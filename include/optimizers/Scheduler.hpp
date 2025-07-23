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

static float onecycle_lr(long long step,
                         long long total_steps,
                         float lr_max      = 3e-4f,
                         float pct_up      = 0.3f,   // 30 % warm‑up
                         float div_factor  = 25.0f)  // lr_min = lr_max/div_factor
{
    long long warmup_steps = (long long)(pct_up * total_steps);
    float lr_min = lr_max / div_factor;
    float lr_final = lr_min / div_factor;         // annihilation target

    if (step < warmup_steps) {
        // lineal: lr_min  →  lr_max
        float progress = (float)step / warmup_steps;
        return lr_min + progress * (lr_max - lr_min);
    }

    long long down_steps = total_steps - warmup_steps;
    long long step_down = step - warmup_steps;
    float progress = (float)step_down / down_steps;

    if (progress < 0.8f) {  // 80 %: fase de descenso suave
        float pct = progress / 0.8f;               // 0‑1
        return lr_max - pct * (lr_max - lr_min);
    } else {                // 20 % final: annihilation
        float pct = (progress - 0.8f) / 0.2f;      // 0‑1
        return lr_min - pct * (lr_min - lr_final);
    }
}
