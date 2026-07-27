#include "ModelSummary.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// This file mirrors the exact parameter shapes used by the real layers in
// gpt/Layers/*.cpp, so the numbers below are not illustrative -- they are
// the same formulas the constructors use:
//
//   Embedding            weights (vocab, dim)                 no bias   -> Embedding.cpp
//   PositionEmbedding     weights (maxSeqLen, dim)              no bias   -> PositionEmbedding.cpp
//   LinearLayer(in, out)  weights (out, in) + bias (out)                  -> Linear.cpp
//   LayerNorms(dim)       gamma (dim) + beta (dim)                        -> LayerNorms.cpp
//   MultiHeadAttention    Wq, Wk, Wv, Wo = 4x LinearLayer(dim, dim)        -> MultiHeadAttention.cpp
//   FeedForward(dim)      fc1: dim->4*dim, fc2: 4*dim->dim                -> MLP.cpp
//   TransformerBlock      LN1 + MHA + (residual, 0 params) + LN2 + MLP
//                         + (residual, 0 params)                         -> Transformer.cpp
//   GPT::lmHead           LinearLayer(dim, vocab)                         -> GPT.cpp
//
// Tensor stores every value as a C++ `double` (see Tensor.hpp), so the
// memory estimate below uses 8 bytes/parameter -- this project has no
// FP32 storage path.
// -----------------------------------------------------------------------

namespace
{
    using u64 = unsigned long long;

    std::string formatWithCommas(u64 value)
    {
        std::string digits = std::to_string(value);
        std::string out;
        int count = 0;
        for (auto it = digits.rbegin(); it != digits.rend(); ++it)
        {
            if (count != 0 && count % 3 == 0) out.push_back(',');
            out.push_back(*it);
            ++count;
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    // Counts display columns, not bytes: UTF-8 continuation bytes (10xxxxxx)
    // don't start a new character, so box-drawing glyphs and the gamma/beta
    // letters (each multi-byte) must still count as a single column or every
    // row containing them drifts out of alignment with the plain-ASCII rows.
    size_t utf8Width(const std::string &s)
    {
        size_t width = 0;
        for (unsigned char c : s)
        {
            if ((c & 0xC0) != 0x80) ++width;
        }
        return width;
    }

    std::string padRight(const std::string &s, size_t width)
    {
        size_t w = utf8Width(s);
        if (w >= width) return s;
        return s + std::string(width - w, ' ');
    }

    std::string padLeft(const std::string &s, size_t width)
    {
        size_t w = utf8Width(s);
        if (w >= width) return s;
        return std::string(width - w, ' ') + s;
    }

    // One row of the layer table.
    struct Row
    {
        std::string layer;
        std::string shape;
        u64 params;
        bool trainable;
        std::string notes;
    };

    u64 linearParams(u64 in, u64 out) { return out * in + out; }
    u64 layerNormParams(u64 dim) { return 2 * dim; }
    u64 embeddingParams(u64 vocab, u64 dim) { return vocab * dim; }
    u64 positionEmbeddingParams(u64 maxSeqLen, u64 dim) { return maxSeqLen * dim; }
    u64 mhaParams(u64 dim) { return 4 * linearParams(dim, dim); }
    u64 mlpParams(u64 dim) { return linearParams(dim, 4 * dim) + linearParams(4 * dim, dim); }

    constexpr size_t W_LAYER = 30;
    constexpr size_t W_SHAPE = 14;
    constexpr size_t W_PARAMS = 13;
    constexpr size_t W_TRAIN = 9;
    constexpr size_t W_NOTES = 34;

    std::string border(const std::string &left, const std::string &mid,
                        const std::string &right)
    {
        std::vector<size_t> widths = {W_LAYER, W_SHAPE, W_PARAMS, W_TRAIN, W_NOTES};
        std::string out = left;
        for (size_t i = 0; i < widths.size(); ++i)
        {
            for (size_t c = 0; c < widths[i] + 2; ++c) out += "\u2500";
            out += (i + 1 < widths.size()) ? mid : right;
        }
        return out;
    }

    std::string rowLine(const Row &r)
    {
        std::ostringstream ss;
        ss << "\u2502 " << padRight(r.layer, W_LAYER) << " \u2502 "
           << padRight(r.shape, W_SHAPE) << " \u2502 "
           << padLeft(formatWithCommas(r.params), W_PARAMS) << " \u2502 "
           << padRight(r.trainable ? "Yes" : "No", W_TRAIN) << " \u2502 "
           << padRight(r.notes, W_NOTES) << " \u2502";
        return ss.str();
    }

    std::string headerLine()
    {
        std::ostringstream ss;
        ss << "\u2502 " << padRight("Layer", W_LAYER) << " \u2502 "
           << padRight("Output Shape", W_SHAPE) << " \u2502 "
           << padLeft("Parameters", W_PARAMS) << " \u2502 "
           << padRight("Trainable", W_TRAIN) << " \u2502 "
           << padRight("Notes", W_NOTES) << " \u2502";
        return ss.str();
    }
}

void ModelSummary::print(const GPTConfig &cfg)
{
    const u64 V = cfg.vocabSize;
    const u64 E = cfg.embedDim;
    const u64 S = cfg.maxSeqLen;
    const u64 H = cfg.numHeads;
    const u64 L = cfg.numLayers;
    const u64 headDim = H ? E / H : 0;

    const std::string seqShape = "(S, " + std::to_string(E) + ")";
    const std::string headShape = "(S, " + std::to_string(V) + ")";

    std::vector<Row> rows;
    rows.push_back({"Token Embedding", seqShape, embeddingParams(V, E), true, "Vocabulary embeddings"});
    rows.push_back({"Position Embedding", seqShape, positionEmbeddingParams(S, E), true, "Learned positions"});

    u64 totalBlockParams = 0;
    for (u64 i = 1; i <= L; ++i)
    {
        const u64 lnP = layerNormParams(E);
        const u64 attnP = mhaParams(E);
        const u64 mlpP = mlpParams(E);
        const u64 blockP = 2 * lnP + attnP + mlpP;
        totalBlockParams += blockP;

        std::ostringstream title;
        title << "Transformer Block " << i;
        rows.push_back({title.str(), seqShape, blockP, true, "Pre-LN residual block"});
        rows.push_back({"\u251C\u2500\u2500 LayerNorm 1", seqShape, lnP, true, "\u03B3 + \u03B2 affine"});

        std::ostringstream attnNote;
        attnNote << H << " heads x " << headDim << " dim, Q/K/V/O";
        rows.push_back({"\u251C\u2500\u2500 MultiHeadAttention", seqShape, attnP, true, attnNote.str()});
        rows.push_back({"\u251C\u2500\u2500 Residual Add", seqShape, 0, false, "x + Attention(LN1(x))"});
        rows.push_back({"\u251C\u2500\u2500 LayerNorm 2", seqShape, lnP, true, "\u03B3 + \u03B2 affine"});
        rows.push_back({"\u251C\u2500\u2500 FeedForward (MLP)", seqShape, mlpP, true, "GELU, hidden " + std::to_string(4 * E)});
        rows.push_back({"\u2514\u2500\u2500 Residual Add", seqShape, 0, false, "x + MLP(LN2(x))"});
    }

    const u64 finalNormP = layerNormParams(E);
    const u64 lmHeadP = linearParams(E, V);
    rows.push_back({"Final LayerNorm", seqShape, finalNormP, true, "\u03B3 + \u03B2 affine"});
    rows.push_back({"LM Head", headShape, lmHeadP, true, "Output projection to vocab"});

    const u64 tokenEmbP = embeddingParams(V, E);
    const u64 posEmbP = positionEmbeddingParams(S, E);
    const u64 totalParams = tokenEmbP + posEmbP + totalBlockParams + finalNormP + lmHeadP;
    const u64 trainableParams = totalParams; // nothing in this model is frozen
    const u64 nonTrainableParams = 0;
    const double memoryBytes = static_cast<double>(totalParams) * sizeof(double);
    const double memoryMB = memoryBytes / (1024.0 * 1024.0);

    std::cout << border("\u250C", "\u252C", "\u2510") << "\n";
    std::cout << headerLine() << "\n";
    std::cout << border("\u251C", "\u253C", "\u2524") << "\n";
    for (const auto &r : rows) std::cout << rowLine(r) << "\n";
    std::cout << border("\u251C", "\u2534", "\u2524") << "\n";

    const std::string topBorder = border("\u250C", "\u252C", "\u2510");
    // UTF-8 byte length of the border differs from its *display* width
    // (each box char is 3 bytes but 1 column), so measure display width by
    // counting code points instead of raw bytes.
    size_t displayWidth = 0;
    for (size_t i = 0; i < topBorder.size();)
    {
        unsigned char c = topBorder[i];
        size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        i += len;
        ++displayWidth;
    }
    const size_t innerWidth = displayWidth - 2;

    auto footerBorder = [&](const std::string &left, const std::string &right) {
        std::string out = left;
        for (size_t i = 0; i < innerWidth; ++i) out += "\u2500";
        out += right;
        return out;
    };

    auto footerRow = [&](const std::string &label, const std::string &value) {
        std::string content = " " + padRight(label, 20) + ": " + value;
        // pad content out to innerWidth, then close it off. Labels/values
        // here are plain ASCII/digits, so byte length == display width.
        size_t contentLen = content.size();
        if (contentLen < innerWidth) content += std::string(innerWidth - contentLen, ' ');
        return "\u2502" + content + "\u2502";
    };

    std::ostringstream mem;
    mem << std::fixed << std::setprecision(2) << memoryMB << " MB (FP64 / double)";

    std::cout << footerRow("Total Parameters", formatWithCommas(totalParams)) << "\n";
    std::cout << footerRow("Trainable", formatWithCommas(trainableParams)) << "\n";
    std::cout << footerRow("Non-Trainable", formatWithCommas(nonTrainableParams)) << "\n";
    std::cout << footerRow("Estimated Memory", mem.str()) << "\n";
    std::cout << footerRow("Embedding Dimension", std::to_string(E)) << "\n";
    std::cout << footerRow("Attention Heads", std::to_string(H)) << "\n";
    std::cout << footerRow("Head Dimension", std::to_string(headDim)) << "\n";
    std::cout << footerRow("Max Sequence Length", std::to_string(S)) << "\n";
    std::cout << footerRow("Vocabulary Size", formatWithCommas(V)) << "\n";
    std::cout << footerBorder("\u2514", "\u2518") << "\n";
}
