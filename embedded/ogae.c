/* Portable float32 reference implementation. No intrinsics or assembly.
 * Keep -ffast-math disabled: nonfinite rejection is part of the API contract.
 * Estimated FFT arithmetic is 5*N*log2(N); measured target timing also includes
 * feature logarithms, sqrt/division, flash access, and all checking below.
 */
#include "ogae.h"
#include <float.h>
#include <math.h>

static int shape_valid(const ogae_model *m) {
    return m && m->samples >= 2 && !(m->samples & (m->samples - 1)) &&
        m->features && !(m->samples % m->features) && m->latent &&
        m->references && m->classes && m->classes <= m->references &&
        m->weight0 && m->bias0 && m->codes && m->reference_classes &&
        m->class_labels && m->twiddle_real && m->twiddle_imag &&
        (!m->hidden || (m->weight1 && m->bias1));
}

size_t ogae_workspace_floats(const ogae_model *m) {
    return shape_valid(m) ? 2u * m->samples + m->features + m->hidden + m->latent : 0;
}

static int finite_array(const float *values, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (!isfinite(values[index])) return 0;
    return 1;
}

ogae_status ogae_validate_model(const ogae_model *m) {
    size_t index, c;
    size_t first;
    if (!shape_valid(m)) return OGAE_INVALID_ARGUMENT;
    first = m->hidden ? m->hidden : m->latent;
    if (!finite_array(m->weight0, first * m->features) ||
        !finite_array(m->bias0, first) ||
        (m->hidden && (!finite_array(m->weight1, (size_t)m->latent * m->hidden) ||
                       !finite_array(m->bias1, m->latent))) ||
        !finite_array(m->codes, (size_t)m->references * m->latent) ||
        !finite_array(m->twiddle_real, m->samples / 2u) ||
        !finite_array(m->twiddle_imag, m->samples / 2u))
        return OGAE_NUMERIC_ERROR;
    for (index = 0; index < m->references; ++index)
        if (m->reference_classes[index] >= m->classes) return OGAE_INVALID_ARGUMENT;
    for (c = 0; c < m->classes; ++c) {
        int represented = 0;
        if (m->class_labels[c] < 0 || (c && m->class_labels[c] <= m->class_labels[c - 1]))
            return OGAE_INVALID_ARGUMENT;
        for (index = 0; index < m->references; ++index)
            if (m->reference_classes[index] == c) represented = 1;
        if (!represented) return OGAE_INVALID_ARGUMENT;
    }
    return OGAE_OK;
}

const float *ogae_features(const ogae_model *m, const float *workspace) {
    return workspace + 2u * m->samples;
}

const float *ogae_latent(const ogae_model *m, const float *workspace) {
    return workspace + 2u * m->samples + m->features + m->hidden;
}

static void fft(const ogae_model *m, float *real, float *imag) {
    unsigned span;
    /* Iterative radix-2 decimation-in-time butterflies. Input was copied into
     * bit-reversed order, so all outputs are naturally ordered. Twiddle tables
     * contain exp(-2*pi*i*k/N) and are generated once in the exported header.
     */
    for (span = 2; span <= m->samples; span <<= 1) {
        unsigned half = span >> 1;
        unsigned stride = m->samples / span;
        unsigned base;
        for (base = 0; base < m->samples; base += span) {
            unsigned offset;
            for (offset = 0; offset < half; ++offset) {
                unsigned a = base + offset;
                unsigned b = a + half;
                unsigned twiddle = offset * stride;
                float wr = m->twiddle_real[twiddle];
                float wi = m->twiddle_imag[twiddle];
                float tr = wr * real[b] - wi * imag[b];
                float ti = wr * imag[b] + wi * real[b];
                float ar = real[a];
                float ai = imag[a];
                real[a] = ar + tr;
                imag[a] = ai + ti;
                real[b] = ar - tr;
                imag[b] = ai - ti;
            }
        }
    }
}

static int affine(const float *input, unsigned inputs, float *output,
                  unsigned outputs, const float *weights, const float *bias,
                  int relu) {
    unsigned row;
    for (row = 0; row < outputs; ++row) {
        unsigned column;
        float value = bias[row];
        const float *weight = weights + (size_t)row * inputs;
        for (column = 0; column < inputs; ++column)
            value += weight[column] * input[column];
        if (!isfinite(value)) return 0;
        output[row] = relu && value < 0.0f ? 0.0f : value;
    }
    return 1;
}

ogae_status ogae_predict(const ogae_model *m, const float *iq,
                         float *workspace, size_t workspace_count,
                         float *scores, int64_t *label) {
    size_t required = ogae_workspace_floats(m);
    float *real, *imag, *features, *hidden, *latent;
    float energy = 0.0f;
    float norm = 0.0f;
    unsigned index, reversed = 0;
    unsigned best = 0;
    if (!required || !iq || !workspace || !scores || !label) return OGAE_INVALID_ARGUMENT;
    if (workspace_count < required) return OGAE_WORKSPACE_TOO_SMALL;
    real = workspace;
    imag = real + m->samples;
    features = imag + m->samples;
    hidden = features + m->features;
    latent = hidden + m->hidden;
    for (index = 0; index < m->samples; ++index) {
        unsigned bit = m->samples >> 1;
        float r = iq[2u * index], i = iq[2u * index + 1u];
        if (!isfinite(r) || !isfinite(i)) return OGAE_NUMERIC_ERROR;
        real[reversed] = r;
        imag[reversed] = i;
        /* Binary carry through the reversed index; no lookup RAM is required. */
        while (reversed & bit) { reversed ^= bit; bit >>= 1; }
        reversed ^= bit;
    }
    fft(m, real, imag);
    for (index = 0; index < m->features; ++index) {
        unsigned width = m->samples / m->features;
        unsigned start = index * width;
        unsigned sample;
        float power = 0.0f;
        for (sample = start; sample < start + width; ++sample)
            power += real[sample] * real[sample] + imag[sample] * imag[sample];
        features[index] = power;
        energy += power;
    }
    if (!isfinite(energy)) return OGAE_NUMERIC_ERROR;
    for (index = 0; index < m->features; ++index)
        features[index] = energy > 0.0f ? log1pf(m->features * (features[index] / energy)) : 0.0f;
    if (m->hidden) {
        if (!affine(features, m->features, hidden, m->hidden, m->weight0, m->bias0, 1) ||
            !affine(hidden, m->hidden, latent, m->latent, m->weight1, m->bias1, 0))
            return OGAE_NUMERIC_ERROR;
    } else if (!affine(features, m->features, latent, m->latent, m->weight0, m->bias0, 0)) {
        return OGAE_NUMERIC_ERROR;
    }
    for (index = 0; index < m->latent; ++index) norm += latent[index] * latent[index];
    if (!isfinite(norm)) return OGAE_NUMERIC_ERROR;
    norm = sqrtf(norm);
    for (index = 0; index < m->latent; ++index)
        latent[index] = norm > 1e-12f ? latent[index] / norm : 0.0f;
    for (index = 0; index < m->classes; ++index) scores[index] = -FLT_MAX;
    for (index = 0; index < m->references; ++index) {
        unsigned component;
        unsigned c = m->reference_classes[index];
        float score = 0.0f;
        const float *code = m->codes + (size_t)index * m->latent;
        if (c >= m->classes) return OGAE_INVALID_ARGUMENT;
        for (component = 0; component < m->latent; ++component)
            score += latent[component] * code[component];
        if (!isfinite(score)) return OGAE_NUMERIC_ERROR;
        if (score > scores[c]) scores[c] = score;
    }
    for (index = 1; index < m->classes; ++index)
        if (scores[index] > scores[best]) best = index;
    *label = m->class_labels[best];
    return OGAE_OK;
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
