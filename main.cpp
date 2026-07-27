#include <iostream>
#include <vector>

#include "gpt/TokenizerLayer/Tokenizer.hpp"
#include "gpt/Layers/Embedding.hpp"
#include "gpt/Layers/Linear.hpp"
#include "gpt/Layers/LayerNorms.hpp"
#include "gpt/GPT.hpp"

using namespace std;

int main()
{
    Tokenizer tokenizer{};
    vector<string> files = {
        "data.txt",
        "data1.txt",
        "data2.txt",
        "data3.txt"
    };

    for(int i = 0; i<files.size();i++){
        tokenizer.tokenized("./Data/"+files[i]);
    }

    tokenizer.displayTokens();
    tokenizer.SaveTokensToFile("./tokens.txt");

    if (tokenizer.totalTokens == 0) {
        cerr << "No tokens found for embedding test." << endl;
        return 1;
    }

    vector<int> token_ids;
    token_ids.reserve(tokenizer.totalTokens);
    for (int id = 0; id < tokenizer.totalTokens; ++id) {
        token_ids.push_back(id);
    }

    const size_t embedding_dim = 256;
    Embedding embedding(tokenizer.totalTokens, embedding_dim);
    Tensor embeddings = embedding.forward(token_ids);

    cout << "Embedding output shape: [" << embeddings.shape()[0] << ", " << embeddings.shape()[1] << "]" << endl;

    LayerNorms layer_norm(embedding_dim);
    Tensor normalized_embeddings = layer_norm.forward(embeddings);

    cout << "Normalized embedding sample for token 0: ";
    for (size_t j = 0; j < min<size_t>(8, embedding_dim); ++j) {
        cout << normalized_embeddings(0, j) << " ";
    }
    cout << endl;

    LinearLayer linear(static_cast<int>(embedding_dim), 4);
    Tensor embedding_grads({embeddings.shape()[0], embeddings.shape()[1]});

    for (size_t i = 0; i < normalized_embeddings.shape()[0]; ++i) {
        Tensor token_vector({embedding_dim});
        for (size_t j = 0; j < embedding_dim; ++j) {
            token_vector(j) = normalized_embeddings(i, j);
        }

        Tensor linear_out = linear.forward(token_vector);
        cout << "Token " << i << " linear output:";
        for (size_t j = 0; j < linear_out.size(); ++j) {
            cout << " " << linear_out(j);
        }
        cout << endl;

        Tensor output_grad({linear_out.size()});
        for (size_t j = 0; j < linear_out.size(); ++j) {
            output_grad(j) = 2.0 * linear_out(j);
        }

        Tensor token_grad = linear.backward(output_grad);
        for (size_t j = 0; j < embedding_dim; ++j) {
            embedding_grads(i, j) = token_grad(j);
        }
    }

    embedding.zero_grad();
    embedding.backward(token_ids, embedding_grads);

    cout << "Embedding gradient sample for token 0: ";
    const Tensor& embed_grad_tensor = embedding.parameters()[0]->grad;
    for (size_t j = 0; j < min<size_t>(8, embedding_dim); ++j) {
        cout << embed_grad_tensor(0, j) << " ";
    }
    cout << endl;

    //=======================================================
    // GPT forward-pass / generation CLI smoke test
    //
    // Weights are randomly initialized (no training loop yet), so the
    // generated text is expected to be gibberish. This only verifies that
    // token embedding -> position embedding -> transformer blocks ->
    // final norm -> LM head -> greedy decoding runs end to end.
    //=======================================================
    cout << "\n--- GPT CLI test (untrained, random weights) ---" << endl;

    const size_t gptEmbedDim = 32;
    const size_t gptMaxSeqLen = 64;
    const size_t gptNumHeads = 4;
    const size_t gptNumLayers = 2;

    GPT gpt(tokenizer.totalTokens, gptEmbedDim, gptMaxSeqLen, gptNumHeads, gptNumLayers);

    cout << "Enter a prompt (characters outside the training vocab are skipped): ";
    string prompt;
    std::getline(cin, prompt);

    vector<int> promptTokens;
    for (char c : prompt) {
        int id = tokenizer.CharToTokenIndex(c);
        if (id >= 0) promptTokens.push_back(id);
    }

    if (promptTokens.empty()) {
        cout << "No recognizable characters in prompt; seeding with token 0 instead." << endl;
        promptTokens.push_back(0);
    }

    if (promptTokens.size() > gptMaxSeqLen) {
        promptTokens.erase(promptTokens.begin(), promptTokens.begin() + (promptTokens.size() - gptMaxSeqLen));
    }

    const size_t newTokenCount = 40;
    cout << "Generating " << newTokenCount << " tokens..." << endl;
    vector<int> output = gpt.generate(promptTokens, newTokenCount);

    cout << "Generated (decoded): \"";
    for (int id : output) {
        cout << tokenizer.TokenIndexToChar(id);
    }
    cout << "\"" << endl;

    return 0;
}