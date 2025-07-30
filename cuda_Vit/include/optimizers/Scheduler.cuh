#pragma once
#include <cmath>
#include <algorithm>

static float cosine_warmup_lr(long long step,
                              long long warmup_steps,
                              long long total_steps,
                              float    lr_max)
{
    if (step < warmup_steps) {
        return lr_max * (float)step / std::max(1LL, warmup_steps);
    }
    float progress = (float)(step - warmup_steps)
                   / std::max(1LL, total_steps - warmup_steps);
    return lr_max * 0.5f * (1.0f + std::cos(M_PI * progress));
}