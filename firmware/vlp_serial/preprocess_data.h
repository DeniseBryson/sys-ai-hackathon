#pragma once

constexpr float kRssScale = 0.849386811f;
// Raw sensor readings run at a slightly lower mean than the clean training data
// (fitted as clean_train.mean() - raw_train.mean()); added before the kRssScale
// divide so it operates on the same unnormalized domain it was fitted on.
constexpr float kRssMeanShift = 0.00492801517f;
