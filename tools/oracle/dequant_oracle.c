// Dumps ggml's own dequantization (PQ2_0 / PTQ1_0) of the first N rows of a tensor as
// little-endian f32, so Rust dequant can be checked bit-exactly. Also emits PTQ1_0 by
// re-quantizing those rows with ggml's quantize_row_ptq1_0_ref.
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char ** argv) {
    // usage: dequant_oracle model.gguf tensor nrows out_prefix
    struct ggml_context * meta = NULL;
    struct gguf_init_params ip = { .no_alloc = true, .ctx = &meta };
    struct gguf_context * g = gguf_init_from_file(argv[1], ip);
    int64_t ti = gguf_find_tensor(g, argv[2]);
    struct ggml_tensor * t = ggml_get_tensor(meta, argv[2]);
    int nrows = atoi(argv[3]); int64_t ne0 = t->ne[0];
    size_t rb = ggml_row_size(t->type, ne0);
    FILE * f = fopen(argv[1], "rb");
    fseek(f, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti), SEEK_SET);
    char * raw = malloc(rb * nrows); fread(raw, 1, rb * nrows, f);
    float * out = malloc(sizeof(float) * ne0 * nrows);
    ggml_get_type_traits(t->type)->to_float(raw, out, ne0 * nrows);
    char p[512];
    snprintf(p, sizeof p, "%s.raw", argv[4]); FILE * o = fopen(p, "wb"); fwrite(raw, 1, rb * nrows, o); fclose(o);
    snprintf(p, sizeof p, "%s.f32", argv[4]); o = fopen(p, "wb"); fwrite(out, 4, ne0 * nrows, o); fclose(o);
    // PTQ1_0 roundtrip of the same values
    size_t rb1 = ggml_row_size(GGML_TYPE_PTQ1_0, ne0);
    char * q1 = malloc(rb1 * nrows);
    ggml_quantize_chunk(GGML_TYPE_PTQ1_0, out, q1, 0, nrows, ne0, NULL);
    float * out1 = malloc(sizeof(float) * ne0 * nrows);
    ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float(q1, out1, ne0 * nrows);
    snprintf(p, sizeof p, "%s.ptq1.raw", argv[4]); o = fopen(p, "wb"); fwrite(q1, 1, rb1 * nrows, o); fclose(o);
    snprintf(p, sizeof p, "%s.ptq1.f32", argv[4]); o = fopen(p, "wb"); fwrite(out1, 4, ne0 * nrows, o); fclose(o);
    printf("type=%d ne0=%lld rows=%d row_bytes=%zu ptq1_row_bytes=%zu\n", t->type, (long long)ne0, nrows, rb, rb1);
    return 0;
}
