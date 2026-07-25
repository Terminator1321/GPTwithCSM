#include<iostream>
#include<vector>

#include "gpt\TokenizerLayer\Tokenizer.hpp"

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
        tokenizer.tokenized("./data/"+files[i]);
    }

    tokenizer.displayTokens();
    tokenizer.SaveTokensToFile("./tokens.txt");
    return 0;
}