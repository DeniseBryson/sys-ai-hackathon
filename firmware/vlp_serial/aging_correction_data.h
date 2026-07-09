#pragma once

// Fresh-install per-channel ceiling (99th percentile of clean training RSS) and the
// peak-tracking tuning validated in notebooks/aging_correction.py against
// vlp_hackathon/aging.py's simulated LED decay. See main.cpp's per-channel
// peak-hold-with-slow-release loop in handle_predict for how these are used.
constexpr int kAgingChannelCount = 9;
constexpr float kAgingReferencePeak[kAgingChannelCount] = {
    0.747199714f, 0.722243249f, 0.742989719f, 0.69120276f, 0.577727556f,
    0.670980752f, 0.755990088f, 0.709912419f, 0.761830389f,
};
constexpr float kAgingRelease = 0.999f;
constexpr float kAgingMaxFactor = 3.0f;
