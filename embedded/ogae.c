/* Portable float32 reference implementation. No intrinsics or assembly.
 * Keep -ffast-math disabled: nonfinite rejection is part of the API contract.
 * All model weights remain immutable; scratch is supplied by the caller.
 */
#include "ogae.h"
#include <float.h>
#include <math.h>

typedef char ogae_requires_float32[(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24) ? 1 : -1];

static int shape_valid(const ogae_model *m) {
    unsigned layer;
    if (!m || m->samples < 2 || (m->samples & (m->samples - 1)) ||
        !m->latent || !m->references || !m->classes || m->classes > m->references ||
        !m->weight0 || !m->bias0 || !m->codes || !m->reference_classes || !m->class_labels)
        return 0;
    if (m->kind == OGAE_SPECTRAL)
        return m->features && !(m->samples % m->features) && m->twiddle_real && m->twiddle_imag &&
            (!m->hidden || (m->weight1 && m->bias1));
    if (m->samples != 512 || !m->conv_channels || !m->conv_weights || !m->conv_biases) return 0;
    if (m->kind == OGAE_TEMPORAL_CONV) {
        if (m->features != 3u * m->samples || (m->conv_layers < 3 || m->conv_layers > 7) ||
            (m->conv_frontend != OGAE_LOCAL_PRODUCTS && m->conv_frontend != OGAE_NORMALIZED_IQ)) return 0;
    } else if (m->kind == OGAE_COHERENT_CONV) {
        if (m->features != 2u * m->samples || m->conv_layers < 3 || m->conv_layers > 5 ||
            !m->coherent_channels || m->coherent_channels > 256 ||
            m->coherent_kernel < 3 || m->coherent_kernel > 65 || !(m->coherent_kernel & 1u) ||
            !m->coherent_real || !m->coherent_imag || !m->power_gain || !m->power_bias) return 0;
    } else return 0;
    for (layer = 0; layer < m->conv_layers; ++layer)
        if (!m->conv_channels[layer] || m->conv_channels[layer] > 256 || !m->conv_weights[layer] || !m->conv_biases[layer])
            return 0;
    return 1;
}

static void conv_buffer_sizes(const ogae_model *m, size_t *a, size_t *b) {
    unsigned layer;
    unsigned first = m->kind == OGAE_COHERENT_CONV;
    *a = first ? (size_t)(m->samples / 2u) * m->coherent_channels : 0;
    *b = 0;
    for (layer = 0; layer < m->conv_layers; ++layer) {
        size_t count = (size_t)(m->samples >> (layer + 1u + first)) * m->conv_channels[layer];
        size_t *maximum = (layer + first) & 1u ? b : a;
        if (count > *maximum) *maximum = count;
    }
}

size_t ogae_workspace_floats(const ogae_model *m) {
    size_t a, b;
    if (!shape_valid(m)) return 0;
    if (m->kind == OGAE_SPECTRAL)
        return 2u * m->samples + m->features + m->hidden + m->latent;
    conv_buffer_sizes(m, &a, &b);
    return m->features + a + b + 2u * m->conv_channels[m->conv_layers - 1u] + m->latent;
}

static int finite_array(const float *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (!isfinite(values[index])) return 0;
    return 1;
}

ogae_status ogae_validate_model(const ogae_model *m) {
    size_t index, c, first;
    if (!shape_valid(m)) return OGAE_INVALID_ARGUMENT;
    if (m->kind == OGAE_SPECTRAL) {
        first = m->hidden ? m->hidden : m->latent;
        if (!finite_array(m->weight0, first * m->features) || !finite_array(m->bias0, first) ||
            (m->hidden && (!finite_array(m->weight1, (size_t)m->latent * m->hidden) || !finite_array(m->bias1, m->latent))) ||
            !finite_array(m->twiddle_real, m->samples / 2u) || !finite_array(m->twiddle_imag, m->samples / 2u))
            return OGAE_NUMERIC_ERROR;
    } else {
        unsigned layer, channels = m->kind == OGAE_COHERENT_CONV ? m->coherent_channels : 3;
        if (m->kind == OGAE_COHERENT_CONV &&
            (!finite_array(m->coherent_real, (size_t)m->coherent_channels * m->coherent_kernel) ||
             !finite_array(m->coherent_imag, (size_t)m->coherent_channels * m->coherent_kernel) ||
             !finite_array(m->power_gain, m->coherent_channels) || !finite_array(m->power_bias, m->coherent_channels)))
            return OGAE_NUMERIC_ERROR;
        for (layer = 0; layer < m->conv_layers; ++layer) {
            if (!finite_array(m->conv_weights[layer], (size_t)m->conv_channels[layer] * channels * 5u) ||
                !finite_array(m->conv_biases[layer], m->conv_channels[layer])) return OGAE_NUMERIC_ERROR;
            channels = m->conv_channels[layer];
        }
        if (!finite_array(m->weight0, (size_t)m->latent * 2u * channels) || !finite_array(m->bias0, m->latent))
            return OGAE_NUMERIC_ERROR;
    }
    if (!finite_array(m->codes, (size_t)m->references * m->latent)) return OGAE_NUMERIC_ERROR;
    for (index = 0; index < m->references; ++index)
        if (m->reference_classes[index] >= m->classes) return OGAE_INVALID_ARGUMENT;
    for (c = 0; c < m->classes; ++c) {
        int represented = 0;
        if (m->class_labels[c] < 0 || (c && m->class_labels[c] <= m->class_labels[c - 1])) return OGAE_INVALID_ARGUMENT;
        for (index = 0; index < m->references; ++index)
            if (m->reference_classes[index] == c) represented = 1;
        if (!represented) return OGAE_INVALID_ARGUMENT;
    }
    return OGAE_OK;
}

const float *ogae_features(const ogae_model *m, const float *workspace) {
    return m->kind != OGAE_SPECTRAL ? workspace : workspace + 2u * m->samples;
}

const float *ogae_latent(const ogae_model *m, const float *workspace) {
    size_t a, b;
    if (m->kind == OGAE_SPECTRAL) return workspace + 2u * m->samples + m->features + m->hidden;
    conv_buffer_sizes(m, &a, &b);
    return workspace + m->features + a + b + 2u * m->conv_channels[m->conv_layers - 1u];
}

static void fft(const ogae_model *m, float *real, float *imag) {
    unsigned span;
    /* Radix-2 DIT: input is bit-reversed and output naturally ordered. Constant
     * twiddles are exp(-2*pi*i*k/N); no runtime trigonometry or lookup RAM. */
    for (span = 2; span <= m->samples; span <<= 1) {
        unsigned half = span >> 1, stride = m->samples / span, base;
        for (base = 0; base < m->samples; base += span) {
            unsigned offset;
            for (offset = 0; offset < half; ++offset) {
                unsigned a = base + offset, b = a + half, twiddle = offset * stride;
                float wr = m->twiddle_real[twiddle], wi = m->twiddle_imag[twiddle];
                float tr = wr * real[b] - wi * imag[b];
                float ti = wr * imag[b] + wi * real[b];
                float ar = real[a], ai = imag[a];
                real[a] = ar + tr; imag[a] = ai + ti;
                real[b] = ar - tr; imag[b] = ai - ti;
            }
        }
    }
}

static int affine(const float *input, unsigned inputs, float *output, unsigned outputs,
                  const float *weights, const float *bias, int relu) {
    unsigned row;
    for (row = 0; row < outputs; ++row) {
        unsigned column;
        float value = bias[row];
        const float *weight = weights + (size_t)row * inputs;
        for (column = 0; column < inputs; ++column) value += weight[column] * input[column];
        if (!isfinite(value)) return 0;
        output[row] = relu && value < 0.0f ? 0.0f : value;
    }
    return 1;
}

/* Shared bank traversal keeps full-IQ and already-encoded queries identical.
 * Bank codes are immutable; strict comparisons retain the first sorted class
 * on equal scores, including the all-zero query produced by a zero encoder. */
static ogae_status match_normalized(const ogae_model *m, const float *latent,
                                     float *scores, int64_t *label) {
    unsigned index, best = 0;
    for (index = 0; index < m->classes; ++index) scores[index] = -FLT_MAX;
    for (index = 0; index < m->references; ++index) {
        unsigned component, c = m->reference_classes[index];
        float score = 0.0f;
        const float *code = m->codes + (size_t)index * m->latent;
        if (c >= m->classes) return OGAE_INVALID_ARGUMENT;
        for (component = 0; component < m->latent; ++component) score += latent[component] * code[component];
        if (!isfinite(score)) return OGAE_NUMERIC_ERROR;
        if (score > scores[c]) scores[c] = score;
    }
    for (index = 1; index < m->classes; ++index) if (scores[index] > scores[best]) best = index;
    *label = m->class_labels[best];
    return OGAE_OK;
}

ogae_status ogae_match_codes(const ogae_model *m, const float *normalized_code,
                             float *scores, int64_t *label) {
    if (!shape_valid(m) || !normalized_code || !scores || !label) return OGAE_INVALID_ARGUMENT;
    if (!finite_array(normalized_code, m->latent)) return OGAE_NUMERIC_ERROR;
    return match_normalized(m, normalized_code, scores, label);
}

static ogae_status finish(const ogae_model *m, float *latent, float *scores, int64_t *label) {
    unsigned index;
    float norm = 0.0f;
    for (index = 0; index < m->latent; ++index) norm += latent[index] * latent[index];
    if (!isfinite(norm)) return OGAE_NUMERIC_ERROR;
    norm = sqrtf(norm);
    for (index = 0; index < m->latent; ++index) latent[index] = norm > 1e-12f ? latent[index] / norm : 0.0f;
    return match_normalized(m, latent, scores, label);
}

static int temporal_frontend(const float *iq, unsigned n, float *features, ogae_conv_frontend kind) {
    unsigned sample;
    unsigned power_offset = kind == OGAE_NORMALIZED_IQ ? 2u * n : 0;
    float energy = 0.0f;
    for (sample = 0; sample < n; ++sample) {
        float real = iq[2u * sample], imag = iq[2u * sample + 1u];
        if (!isfinite(real) || !isfinite(imag)) return 0;
        features[power_offset + sample] = real * real + imag * imag;
        energy += features[power_offset + sample];
    }
    if (!isfinite(energy)) return 0;
    if (kind == OGAE_NORMALIZED_IQ) {
        /* Match the stored NumPy frontend order, including its treatment of
         * subnormal energy: RMS is sqrt(E/N), followed by I/RMS and Q/RMS. */
        float norm = sqrtf(energy / n);
        for (sample = 0; sample < n; ++sample) {
            features[sample] = norm > 0.0f ? iq[2u * sample] / norm : 0.0f;
            features[n + sample] = norm > 0.0f ? iq[2u * sample + 1u] / norm : 0.0f;
            features[2u * n + sample] = energy > 0.0f ? n * (features[2u * n + sample] / energy) : 0.0f;
        }
    } else {
        for (sample = 0; sample < n; ++sample) {
            unsigned previous = sample ? sample - 1u : n - 1u;
            float real = iq[2u * sample], imag = iq[2u * sample + 1u];
            float pr = iq[2u * previous], pi = iq[2u * previous + 1u];
            features[n + sample] = real * pr + imag * pi;
            features[2u * n + sample] = imag * pr - real * pi;
        }
        for (sample = 0; sample < 3u * n; ++sample)
            features[sample] = energy > 0.0f ? n * (features[sample] / energy) : 0.0f;
    }
    return 1;
}
static int coherent_frontend(const float *iq, unsigned n, float *features) {
    unsigned sample;
    float energy = 0.0f, rms;
    for (sample = 0; sample < n; ++sample) {
        float real = iq[2u * sample], imag = iq[2u * sample + 1u];
        if (!isfinite(real) || !isfinite(imag)) return 0;
        energy += real * real + imag * imag;
    }
    if (!isfinite(energy)) return 0;
    rms = sqrtf(energy / n);
    for (sample = 0; sample < n; ++sample) {
        features[sample] = rms > 0.0f ? iq[2u * sample] / rms : 0.0f;
        features[n + sample] = rms > 0.0f ? iq[2u * sample + 1u] / rms : 0.0f;
    }
    return 1;
}

static int coherent_power(const ogae_model *m, const float *input, float *output) {
    unsigned channel, position, length = m->samples, output_length = length / 2u;
    for (channel = 0; channel < m->coherent_channels; ++channel) {
        const float *wr = m->coherent_real + (size_t)channel * m->coherent_kernel;
        const float *wi = m->coherent_imag + (size_t)channel * m->coherent_kernel;
        for (position = 0; position < output_length; ++position) {
            unsigned tap;
            unsigned start = 2u * position + length - m->coherent_kernel / 2u;
            float real = 0.0f, imag = 0.0f, power, value;
            /* Coherent complex accumulation precedes the nonlinear detector.
             * No complex bias is used: it would break global-phase equivariance.
             * First-layer BatchNorm gain/bias act AFTER magnitude-squaring and
             * remain explicit, including negative gains, before ReLU. */
            for (tap = 0; tap < m->coherent_kernel; ++tap) {
                unsigned sample = (start + tap) & (length - 1u);
                float xr = input[sample], xi = input[length + sample];
                real += wr[tap] * xr; real -= wi[tap] * xi;
                imag += wi[tap] * xr; imag += wr[tap] * xi;
            }
            if (!isfinite(real) || !isfinite(imag)) return 0;
            power = real * real + imag * imag;
            if (!isfinite(power)) return 0;
            value = power * m->power_gain[channel] + m->power_bias[channel];
            if (!isfinite(value)) return 0;
            output[(size_t)channel * output_length + position] = value > 0.0f ? value : 0.0f;
        }
    }
    return 1;
}
static int conv_relu(const float *input, unsigned inputs, unsigned length, float *output,
                     unsigned outputs, const float *weights, const float *bias) {
    unsigned row, position, output_length = length >> 1, mask = length - 1u;
    for (row = 0; row < outputs; ++row) {
        for (position = 0; position < output_length; ++position) {
            unsigned channel;
            unsigned a = (2u * position + length - 2u) & mask;
            unsigned b = (a + 1u) & mask, c = (a + 2u) & mask;
            unsigned d = (a + 3u) & mask, e = (a + 4u) & mask;
            float value = bias[row];
            /* The five taps are explicit so address wrapping is shared across
             * input channels. This is ordinary C, with no intrinsics or im2col. */
            for (channel = 0; channel < inputs; ++channel) {
                const float *x = input + (size_t)channel * length;
                const float *w = weights + ((size_t)row * inputs + channel) * 5u;
                value += w[0] * x[a]; value += w[1] * x[b]; value += w[2] * x[c];
                value += w[3] * x[d]; value += w[4] * x[e];
            }
            if (!isfinite(value)) return 0;
            output[(size_t)row * output_length + position] = value > 0.0f ? value : 0.0f;
        }
    }
    return 1;
}

static ogae_status predict_conv(const ogae_model *m, const float *iq, float *workspace,
                                float *scores, int64_t *label) {
    size_t a_size, b_size;
    float *a, *b, *pool, *latent;
    const float *input = workspace;
    unsigned layer, channel, length = m->samples, channels = 3;
    unsigned first = m->kind == OGAE_COHERENT_CONV;
    conv_buffer_sizes(m, &a_size, &b_size);
    a = workspace + m->features; b = a + a_size; pool = b + b_size;
    latent = pool + 2u * m->conv_channels[m->conv_layers - 1u];
    if (first) {
        if (!coherent_frontend(iq, m->samples, workspace) || !coherent_power(m, workspace, a)) return OGAE_NUMERIC_ERROR;
        input = a; channels = m->coherent_channels; length >>= 1;
    } else if (!temporal_frontend(iq, m->samples, workspace, m->conv_frontend)) return OGAE_NUMERIC_ERROR;
    for (layer = 0; layer < m->conv_layers; ++layer) {
        float *output = (layer + first) & 1u ? b : a;
        if (!conv_relu(input, channels, length, output, m->conv_channels[layer],
                        m->conv_weights[layer], m->conv_biases[layer])) return OGAE_NUMERIC_ERROR;
        input = output; channels = m->conv_channels[layer]; length >>= 1;
    }
    for (channel = 0; channel < channels; ++channel) {
        unsigned sample;
        float sum = 0.0f, maximum = input[(size_t)channel * length];
        for (sample = 0; sample < length; ++sample) {
            float value = input[(size_t)channel * length + sample];
            sum += value;
            if (value > maximum) maximum = value;
        }
        pool[channel] = sum / length; pool[channels + channel] = maximum;
    }
    if (!affine(pool, 2u * channels, latent, m->latent, m->weight0, m->bias0, 0)) return OGAE_NUMERIC_ERROR;
    return finish(m, latent, scores, label);
}

ogae_status ogae_predict(const ogae_model *m, const float *iq, float *workspace,
                         size_t workspace_count, float *scores, int64_t *label) {
    size_t required = ogae_workspace_floats(m);
    float *real, *imag, *features, *hidden, *latent;
    float energy = 0.0f;
    unsigned index, reversed = 0;
    if (!required || !iq || !workspace || !scores || !label) return OGAE_INVALID_ARGUMENT;
    if (workspace_count < required) return OGAE_WORKSPACE_TOO_SMALL;
    if (m->kind != OGAE_SPECTRAL) return predict_conv(m, iq, workspace, scores, label);
    real = workspace; imag = real + m->samples; features = imag + m->samples;
    hidden = features + m->features; latent = hidden + m->hidden;
    for (index = 0; index < m->samples; ++index) {
        unsigned bit = m->samples >> 1;
        float r = iq[2u * index], i = iq[2u * index + 1u];
        if (!isfinite(r) || !isfinite(i)) return OGAE_NUMERIC_ERROR;
        real[reversed] = r; imag[reversed] = i;
        /* Binary carry through a bit-reversed index, requiring no lookup RAM. */
        while (reversed & bit) { reversed ^= bit; bit >>= 1; }
        reversed ^= bit;
    }
    fft(m, real, imag);
    for (index = 0; index < m->features; ++index) {
        unsigned width = m->samples / m->features, start = index * width, sample;
        float power = 0.0f;
        for (sample = start; sample < start + width; ++sample)
            power += real[sample] * real[sample] + imag[sample] * imag[sample];
        features[index] = power; energy += power;
    }
    if (!isfinite(energy)) return OGAE_NUMERIC_ERROR;
    for (index = 0; index < m->features; ++index)
        features[index] = energy > 0.0f ? log1pf(m->features * (features[index] / energy)) : 0.0f;
    if (m->hidden) {
        if (!affine(features, m->features, hidden, m->hidden, m->weight0, m->bias0, 1) ||
            !affine(hidden, m->hidden, latent, m->latent, m->weight1, m->bias1, 0)) return OGAE_NUMERIC_ERROR;
    } else if (!affine(features, m->features, latent, m->latent, m->weight0, m->bias0, 0)) return OGAE_NUMERIC_ERROR;
    return finish(m, latent, scores, label);
}

ogae_status ogae_aligned_predict(const ogae_aligned_model *m, const float *iq,
                                 float *scores, int64_t *label) {
    unsigned sample, reference, best = 0;
    float norm = 0.0f;
    if (!m || !m->samples || !m->references || !m->classes ||
        m->classes > m->references || !m->templates || !m->reference_classes ||
        !m->class_labels || !iq || !scores || !label) return OGAE_INVALID_ARGUMENT;
    for (sample = 0; sample < m->samples; ++sample) {
        float real = iq[2u * sample], imag = iq[2u * sample + 1u];
        if (!isfinite(real) || !isfinite(imag)) return OGAE_NUMERIC_ERROR;
        norm += real * real + imag * imag;
    }
    if (!isfinite(norm)) return OGAE_NUMERIC_ERROR;
    norm = sqrtf(norm);
    for (reference = 0; reference < m->classes; ++reference) scores[reference] = -FLT_MAX;
    for (reference = 0; reference < m->references; ++reference) {
        float real = 0.0f, imag = 0.0f, score;
        unsigned c = m->reference_classes[reference];
        const float *template_iq = m->templates + (size_t)reference * m->samples * 2u;
        if (c >= m->classes) return OGAE_INVALID_ARGUMENT;
        for (sample = 0; sample < m->samples; ++sample) {
            float xr = iq[2u * sample], xi = iq[2u * sample + 1u];
            float tr = template_iq[2u * sample], ti = template_iq[2u * sample + 1u];
            real += xr * tr + xi * ti;
            imag += xi * tr - xr * ti;
        }
        score = sqrtf(real * real + imag * imag);
        if (!isfinite(score)) return OGAE_NUMERIC_ERROR;
        score = norm > 1e-12f ? score / norm : 0.0f;
        if (score > scores[c]) scores[c] = score;
    }
    for (reference = 1; reference < m->classes; ++reference)
        if (scores[reference] > scores[best]) best = reference;
    *label = m->class_labels[best];
    return OGAE_OK;
}
