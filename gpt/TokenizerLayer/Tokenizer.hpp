#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

class Tokenizer
{
private:
    std::vector<char> tokens;

public:
    int totalTokens = 0;

    Tokenizer();
    explicit Tokenizer(const std::string& filePath);

    void tokenized(const std::string& filePath);

    void displayTokens();

    void SaveTokensToFile(const std::string& outputFilePath);

    std::vector<char> LoadTokensFromFile(const std::string& inputFilePath);

    int CharToTokenIndex(char c) const;

    char TokenIndexToChar(int index);
};

#endif 