#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "gpt/GPT.hpp"
#include "gpt/NameSpaces/ActivationFunction.hpp"
#include "gpt/NameSpaces/LOSS.hpp"
#include "gpt/Optimizers/Adam.hpp"
#include "gpt/TokenizerLayer/Tokenizer.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

std::vector<int> loadTokenIds(const Tokenizer& tokenizer, const std::vector<std::string>& files)
{
    std::vector<int> token_ids;
    token_ids.reserve(4096);

    for (const std::string& file : files) {
        std::ifstream input("./Data/" + file);
        if (!input.is_open()) {
            std::cerr << "Warning: could not open " << file << std::endl;
            continue;
        }

        char c = '\0';
        while (input.get(c)) {
            const int id = tokenizer.CharToTokenIndex(c);
            if (id >= 0) {
                token_ids.push_back(id);
            }
        }
    }

    return token_ids;
}

std::vector<std::vector<int>> makeWindowSequences(const std::vector<int>& token_ids, size_t context_window)
{
    std::vector<std::vector<int>> sequences;
    if (token_ids.size() <= context_window + 1) {
        return sequences;
    }

    for (size_t i = 0; i + context_window + 1 < token_ids.size(); ++i) {
        std::vector<int> seq(token_ids.begin() + i, token_ids.begin() + i + context_window + 1);
        sequences.push_back(std::move(seq));
    }

    return sequences;
}

double sequenceLoss(GPT& gpt, size_t vocab_size, const std::vector<int>& inputs, const std::vector<int>& targets)
{
    Tensor logits = gpt.forward(inputs);
    const size_t seq_len = inputs.size();
    double total_loss = 0.0;

    for (size_t s = 0; s < seq_len; ++s) {
        Tensor row({vocab_size});
        for (size_t j = 0; j < vocab_size; ++j) {
            row(j) = logits(s, j);
        }

        Tensor probs = Activation::softmaxForward(row);
        total_loss += Loss::crossEntropyForward(probs, static_cast<size_t>(targets[s]));
    }

    return total_loss / static_cast<double>(seq_len);
}

} 

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    Tokenizer tokenizer{};
    const std::vector<std::string> files = {"data.txt", "data1.txt", "data2.txt", "data3.txt"};

    for (const std::string& file : files) {
        tokenizer.tokenized("./Data/" + file);
    }

    if (tokenizer.totalTokens <= 1) {
        std::cerr << "No useful vocabulary was found in the training data." << std::endl;
        return 1;
    }

    std::vector<int> token_ids = loadTokenIds(tokenizer, files);
    if (token_ids.size() < 16) {
        std::cerr << "Not enough tokens to train a meaningful sequence model." << std::endl;
        return 1;
    }

    const size_t context_window = 16;
    const size_t gpt_embed_dim = 16;
    const size_t gpt_max_seq_len = context_window;
    const size_t gpt_num_heads = 1;
    const size_t gpt_num_layers = 1;
    const int epochs = 20;

    std::vector<std::vector<int>> sequences = makeWindowSequences(token_ids, context_window);
    if (sequences.empty()) {
        std::cerr << "No training windows could be built from the token stream." << std::endl;
        return 1;
    }

    const size_t training_window_cap = 64;
    std::mt19937 shuffle_rng(42);
    std::shuffle(sequences.begin(), sequences.end(), shuffle_rng);
    if (sequences.size() > training_window_cap) {
        sequences.erase(sequences.begin() + training_window_cap, sequences.end());
    }

    const size_t val_count = std::max<size_t>(1, sequences.size() / 10);
    const size_t train_count = sequences.size() - val_count;
    std::vector<std::vector<int>> train_sequences(sequences.begin(), sequences.begin() + train_count);
    std::vector<std::vector<int>> val_sequences(sequences.begin() + train_count, sequences.end());

    GPT gpt(tokenizer.totalTokens, gpt_embed_dim, gpt_max_seq_len, gpt_num_heads, gpt_num_layers);

    std::cout << "\n--- Model Summary ---" << std::endl;
    gpt.summary();

    Adam optimizer(1e-3);
    std::cout << "\nTraining progress" << std::endl;
    std::cout << "-----------------" << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_train_loss = 0.0;

        for (const auto& seq : train_sequences) {
            std::vector<int> inputs(seq.begin(), seq.end() - 1);
            std::vector<int> targets(seq.begin() + 1, seq.end());

            std::vector<Parameter*> params = gpt.parameters();
            for (Parameter* param : params) {
                param->zero_grad();
            }

            const double batch_loss = gpt.trainOnBatch(inputs, targets);
            optimizer.step(params);
            epoch_train_loss += batch_loss;
        }

        epoch_train_loss /= static_cast<double>(std::max<size_t>(1, train_sequences.size()));

        double epoch_val_loss = 0.0;
        if (!val_sequences.empty()) {
            for (const auto& seq : val_sequences) {
                std::vector<int> inputs(seq.begin(), seq.end() - 1);
                std::vector<int> targets(seq.begin() + 1, seq.end());
                epoch_val_loss += sequenceLoss(gpt, tokenizer.totalTokens, inputs, targets);
            }
            epoch_val_loss /= static_cast<double>(val_sequences.size());
        } else {
            epoch_val_loss = std::numeric_limits<double>::quiet_NaN();
        }

        const size_t progress_width = 20;
        const size_t filled = static_cast<size_t>((static_cast<double>(epoch) / static_cast<double>(epochs)) * progress_width);
        const std::string bar(progress_width, '-');
        std::string progress = std::string(filled, '#') + bar.substr(filled);

        std::cout << "epoch " << std::setw(2) << epoch << "/" << epochs
                  << " | train_loss=" << std::fixed << std::setprecision(4) << epoch_train_loss
                  << " | val_loss=";
        if (std::isnan(epoch_val_loss)) {
            std::cout << "n/a";
        } else {
            std::cout << std::fixed << std::setprecision(4) << epoch_val_loss;
        }
        std::cout << " | " << progress << std::endl;
    }

    std::cout << "\nTraining complete." << std::endl;

    std::cout << "\n--- CLI Test Mode ---" << std::endl;
    std::cout << "Enter a prompt (characters outside the training vocabulary are skipped)." << std::endl;
    std::cout << "Press Enter on an empty prompt to exit." << std::endl;

    while (true) {
        std::cout << "\nPrompt: ";
        std::string prompt;
        std::getline(std::cin, prompt);

        if (prompt.empty()) {
            break;
        }

        std::vector<int> prompt_tokens;
        for (char c : prompt) {
            const int id = tokenizer.CharToTokenIndex(c);
            if (id >= 0) {
                prompt_tokens.push_back(id);
            }
        }

        if (prompt_tokens.empty()) {
            std::cout << "No recognizable characters in the prompt; using token 0 as a fallback." << std::endl;
            prompt_tokens.push_back(0);
        }

        if (prompt_tokens.size() > gpt_max_seq_len) {
            prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + (prompt_tokens.size() - gpt_max_seq_len));
        }

        const size_t new_token_count = 24;
        const float temperature = 0.8f;
        const int top_k = 40;
        std::vector<int> generated = gpt.generate(prompt_tokens, new_token_count, temperature, top_k);

        std::cout << "Generated: ";
        for (int token : generated) {
            std::cout << tokenizer.TokenIndexToChar(token);
        }
        std::cout << std::endl;
    }

    return 0;
}