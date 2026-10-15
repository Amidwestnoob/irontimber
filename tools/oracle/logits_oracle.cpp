// Reference oracle: runs llama-prism (CPU) on token ids, prints per-position top-k logits
// and greedy continuation as JSON. Usage: logits_oracle model.gguf n_predict id id id ...
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    int n_pred = atoi(argv[2]);
    std::vector<llama_token> toks; for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 8; cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(m, cp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    printf("{\"positions\":[");
    // feed one token at a time so every position's logits are available
    std::vector<llama_token> all = toks; int n_prompt = toks.size();
    for (int i = 0; i < n_prompt + n_pred; i++) {
        llama_batch b = llama_batch_get_one(&all[i], 1);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        const float * lg = llama_get_logits_ith(ctx, -1);
        std::vector<int> idx(nv); for (int j = 0; j < nv; j++) idx[j] = j;
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int c){ return lg[a] > lg[c]; });
        printf("%s{\"pos\":%d,\"top\":[", i ? "," : "", i);
        for (int k = 0; k < 10; k++) printf("%s[%d,%.6f]", k ? "," : "", idx[k], lg[idx[k]]);
        printf("]}");
        if (i >= n_prompt - 1 && i < n_prompt + n_pred - 1) all.push_back(idx[0]);
        fflush(stdout);
    }
    printf("],\"greedy\":[");
    for (int i = n_prompt; i < (int)all.size(); i++) printf("%s%d", i > n_prompt ? "," : "", all[i]);
    printf("]}\n");
    llama_free(ctx); llama_model_free(m); return 0;
}
