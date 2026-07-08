#include <cmath>
#include <cstdint>
#include <cstring>

#include "pico/stdio.h"
#include "pico/stdlib.h"

#include "model_data.h"
#include "preprocess_data.h"
#include "target_scale_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr uint8_t kRequestMagic[4] = {'V', 'L', 'P', '1'};
constexpr uint8_t kPredictMagic[4] = {'V', 'L', 'R', '1'};
constexpr uint8_t kInfoMagic[4] = {'V', 'L', 'I', '1'};
constexpr uint8_t kCmdPredict = 1;
constexpr uint8_t kCmdInfo = 2;
constexpr uint8_t kStatusOk = 0;
constexpr uint8_t kStatusBadMagic = 1;
constexpr uint8_t kStatusBadCommand = 2;
constexpr uint8_t kStatusBadFeatureCount = 3;
constexpr uint8_t kStatusInvokeFailed = 4;
constexpr uint32_t kReadTimeoutUs = 5000000;
constexpr size_t kTensorArenaBytes = 80 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaBytes];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool read_exact(uint8_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const int c = getchar_timeout_us(kReadTimeoutUs);
    if (c == PICO_ERROR_TIMEOUT) {
      return false;
    }
    dst[i] = static_cast<uint8_t>(c & 0xff);
  }
  return true;
}

void write_exact(const uint8_t* src, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    putchar_raw(src[i]);
  }
  stdio_flush();
}

void put_u16_le(uint8_t* p, uint16_t value) {
  p[0] = static_cast<uint8_t>(value & 0xffu);
  p[1] = static_cast<uint8_t>((value >> 8) & 0xffu);
}

void put_u32_le(uint8_t* p, uint32_t value) {
  p[0] = static_cast<uint8_t>(value & 0xffu);
  p[1] = static_cast<uint8_t>((value >> 8) & 0xffu);
  p[2] = static_cast<uint8_t>((value >> 16) & 0xffu);
  p[3] = static_cast<uint8_t>((value >> 24) & 0xffu);
}

uint16_t get_u16_le(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) |
         (static_cast<uint16_t>(p[1]) << 8);
}

float get_f32_le(const uint8_t* p) {
  uint32_t bits = static_cast<uint32_t>(p[0]) |
                  (static_cast<uint32_t>(p[1]) << 8) |
                  (static_cast<uint32_t>(p[2]) << 16) |
                  (static_cast<uint32_t>(p[3]) << 24);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void put_f32_le(uint8_t* p, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  put_u32_le(p, bits);
}

int feature_count() {
  if (input == nullptr || input->dims == nullptr) {
    return 0;
  }
  int count = 1;
  for (int i = 0; i < input->dims->size; ++i) {
    count *= input->dims->data[i];
  }
  return count;
}

void send_predict_status(uint8_t status, float x_cm, float y_cm, uint32_t invoke_us) {
  uint8_t response[20] = {0};
  std::memcpy(response, kPredictMagic, 4);
  response[4] = status;
  put_f32_le(response + 8, x_cm);
  put_f32_le(response + 12, y_cm);
  put_u32_le(response + 16, invoke_us);
  write_exact(response, sizeof(response));
}

void send_info(uint8_t status) {
  uint8_t response[16] = {0};
  std::memcpy(response, kInfoMagic, 4);
  response[4] = status;
  put_u16_le(response + 5, static_cast<uint16_t>(feature_count()));
  put_u32_le(response + 8, g_vlp_model_data_len);
  put_u32_le(response + 12, static_cast<uint32_t>(kTensorArenaBytes));
  write_exact(response, sizeof(response));
}

bool initialize_model() {
  model = tflite::GetModel(g_vlp_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return false;
  }

  // Add operators here if the model architecture changes.
  static tflite::MicroMutableOpResolver<3> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return false;
  }
  if (resolver.AddRelu() != kTfLiteOk) {
    return false;
  }
  if (resolver.AddLogistic() != kTfLiteOk) {
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaBytes);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input == nullptr || output == nullptr) {
    return false;
  }
  const bool float_io =
      input->type == kTfLiteFloat32 && output->type == kTfLiteFloat32;
  const bool int8_io =
      input->type == kTfLiteInt8 && output->type == kTfLiteInt8;
  if (!float_io && !int8_io) {
    return false;
  }
  if (output->dims == nullptr || output->dims->size < 1) {
    return false;
  }
  return true;
}

void handle_predict(uint16_t n_features) {
  const int expected = feature_count();
  if (n_features != expected) {
    // Drain the payload so the next request stays aligned.
    uint8_t discard[4];
    for (uint16_t i = 0; i < n_features; ++i) {
      if (!read_exact(discard, sizeof(discard))) {
        return;
      }
    }
    send_predict_status(kStatusBadFeatureCount, 0.0f, 0.0f, 0);
    return;
  }

  for (int i = 0; i < expected; ++i) {
    uint8_t raw[4];
    if (!read_exact(raw, sizeof(raw))) {
      return;
    }
    const float rss = get_f32_le(raw);
    const float normalized = rss / kRssScale;
    if (input->type == kTfLiteFloat32) {
      input->data.f[i] = normalized;
    } else {
      const float q =
          std::round(normalized / input->params.scale) + input->params.zero_point;
      const int32_t clamped = static_cast<int32_t>(
          q < -128.0f ? -128.0f : (q > 127.0f ? 127.0f : q));
      input->data.int8[i] = static_cast<int8_t>(clamped);
    }
  }

  const uint32_t start_us = time_us_32();
  const TfLiteStatus status = interpreter->Invoke();
  const uint32_t invoke_us = time_us_32() - start_us;
  if (status != kTfLiteOk) {
    send_predict_status(kStatusInvokeFailed, 0.0f, 0.0f, invoke_us);
    return;
  }

  float x_norm = 0.0f;
  float y_norm = 0.0f;
  if (output->type == kTfLiteFloat32) {
    x_norm = output->data.f[0];
    y_norm = output->data.f[1];
  } else {
    x_norm =
        (static_cast<int32_t>(output->data.int8[0]) - output->params.zero_point) *
        output->params.scale;
    y_norm =
        (static_cast<int32_t>(output->data.int8[1]) - output->params.zero_point) *
        output->params.scale;
  }
  const float x_cm = kTargetXMinCm + x_norm * kTargetXRangeCm;
  const float y_cm = kTargetYMinCm + y_norm * kTargetYRangeCm;
  send_predict_status(kStatusOk, x_cm, y_cm, invoke_us);
}

}  // namespace

int main() {
  stdio_init_all();

  if (!initialize_model()) {
    // Keep USB CDC free for the binary host protocol.
    while (true) {
      tight_loop_contents();
    }
  }

  while (true) {
    uint8_t header[8];
    if (!read_exact(header, sizeof(header))) {
      continue;
    }
    if (std::memcmp(header, kRequestMagic, 4) != 0) {
      send_predict_status(kStatusBadMagic, 0.0f, 0.0f, 0);
      continue;
    }

    const uint8_t command = header[4];
    const uint16_t n_features = get_u16_le(header + 5);
    if (command == kCmdPredict) {
      handle_predict(n_features);
    } else if (command == kCmdInfo) {
      send_info(kStatusOk);
    } else {
      send_predict_status(kStatusBadCommand, 0.0f, 0.0f, 0);
    }
  }
}
