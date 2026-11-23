// mistral4/vision/m4_vision.cpp - strata-m4's image encoder helper: Mistral Small 4's Pixtral tower and projector
// (the mmproj GGUF) through llama.cpp's libmtmd, as a resident process.
//
//   m4-vision --mmproj mmproj.gguf --model model.gguf [--cpu] [--threads N] [--max-tokens N]
//   stdin:  ENC <image file> <output .f32 file>
//   stdout: OK {"n_tokens": N, "n_embd": E, "pre": [ids], "post": [ids]}   (or ERR <reason>)
//
// The output file holds the image's N embedding rows (f32, N x E).  `pre` / `post` are the text tokens mtmd puts
// right before / after the image rows for this model (e.g. [IMG_END] after), so the caller can lay out the prompt
// exactly as llama.cpp would.  The model GGUF is opened vocab-only (hyper-parameters and vocabulary, no weights).
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string ids_json(const llama_token* t, size_t n) {
    std::string s = "[";
    for (size_t i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(t[i]);
    return s + "]";
}

int main(int argc, char** argv) {
    std::string mmproj, model;
    bool gpu = true;
    int threads = 16, max_tokens = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--cpu") gpu = false;
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: m4-vision --mmproj FILE --model FILE [--cpu] [--threads N] [--max-tokens N]\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) {
        std::fprintf(stderr, "m4-vision: cannot open %s\n", model.c_str());
        return 1;
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.n_threads = threads;
    cp.print_timings = false;
    cp.warmup = true;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), lm, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::fprintf(stderr, "m4-vision: %s is not a vision projector mtmd can load\n", mmproj.c_str());
        return 1;
    }
    const int n_embd = llama_model_n_embd(lm);
    std::printf("READY {\"n_embd\": %d}\n", n_embd);
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd, img, out;
        is >> cmd >> img >> out;
        if (cmd == "QUIT") break;
        if (cmd != "ENC" || img.empty() || out.empty()) {
            std::printf("ERR expected: ENC <image> <out.f32>\n");
            std::fflush(stdout);
            continue;
        }
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) {
            std::printf("ERR cannot read %s\n", img.c_str());
            std::fflush(stdout);
            continue;
        }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm = bw.bitmap;
        if (mtmd_tokenize(ctx, chunks, &txt, &bm, 1) != 0) {
            std::printf("ERR tokenize failed\n");
            std::fflush(stdout);
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bw.bitmap);
            continue;
        }
        std::string pre = "[]", post = "[]";
        size_t n_img = 0;
        bool seen = false, ok = true;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
            const auto ty = mtmd_input_chunk_get_type(c);
            if (ty == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
                (seen ? post : pre) = ids_json(t, n);
            } else if (ty == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (mtmd_encode_chunk(ctx, c) != 0) { ok = false; break; }
                n_img = mtmd_input_chunk_get_n_tokens(c);
                const float* e = mtmd_get_output_embd(ctx);
                FILE* f = std::fopen(out.c_str(), "wb");
                if (!f || std::fwrite(e, sizeof(float), n_img * (size_t) n_embd, f) != n_img * (size_t) n_embd) ok = false;
                if (f) std::fclose(f);
                seen = true;
            }
        }
        if (ok && seen) std::printf("OK {\"n_tokens\": %zu, \"n_embd\": %d, \"pre\": %s, \"post\": %s}\n", n_img, n_embd, pre.c_str(), post.c_str());
        else std::printf("ERR encode failed\n");
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
    }
    mtmd_free(ctx);
    llama_model_free(lm);
    return 0;
}
