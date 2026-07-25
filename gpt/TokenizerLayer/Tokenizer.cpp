#include "Tokenizer.hpp"

Tokenizer::Tokenizer() {}

Tokenizer::Tokenizer(const std::string& filePath)
{
    tokenized(filePath);
}

void Tokenizer::tokenized(const std::string& filePath)
{
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    char c;

    while (file.get(c))
    {
        totalTokens++;

        bool found = false;

        for (char token : tokens)
        {
            if (token == c)
            {
                totalTokens--;
                found = true;
                break;
            }
        }

        if (!found)
            tokens.push_back(c);
    }

    file.close();
}

void Tokenizer::displayTokens()
{
    std::cout << "Unique Tokens: ";

    for (char token : tokens)
        std::cout << token << ' ';

    std::cout << std::endl;
}

void Tokenizer::SaveTokensToFile(const std::string& outputFilePath)
{
    std::ofstream outputFile(outputFilePath);

    if (!outputFile.is_open())
    {
        std::cerr << "Error opening output file: " << outputFilePath << std::endl;
        return;
    }

    for (char token : tokens)
        outputFile << token << '\n';

    outputFile.close();
}

std::vector<char> Tokenizer::LoadTokensFromFile(const std::string& inputFilePath)
{
    std::vector<char> loadedTokens;

    std::ifstream inputFile(inputFilePath);

    if (!inputFile.is_open())
    {
        std::cerr << "Error opening input file: " << inputFilePath << std::endl;
        return loadedTokens;
    }

    char token;

    while (inputFile.get(token))
        loadedTokens.push_back(token);

    inputFile.close();

    return loadedTokens;
}

int Tokenizer::CharToTokenIndex(char c)
{
    for (size_t i = 0; i < tokens.size(); i++)
    {
        if (tokens[i] == c)
            return static_cast<int>(i);
    }

    return -1;
}

char Tokenizer::TokenIndexToChar(int index)
{
    if (index >= 0 && index < static_cast<int>(tokens.size()))
        return tokens[index];

    return '\0';
}