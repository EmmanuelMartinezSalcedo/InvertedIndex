#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <filesystem>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;

#define MAX_WORD_LENGTH 32
#define MAX_WORDS 1000000

__global__ void processWords(char *words, int *fileIds, int wordCount, char *uniqueWords, int *wordPresence, int uniqueWordCount, int fileId) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < wordCount) {
        char word[MAX_WORD_LENGTH];
        strncpy(word, &words[idx * MAX_WORD_LENGTH], MAX_WORD_LENGTH);
        word[MAX_WORD_LENGTH - 1] = '\0';
        
        for (int i = 0; i < uniqueWordCount; i++) {
            if (strncmp(word, &uniqueWords[i * MAX_WORD_LENGTH], MAX_WORD_LENGTH) == 0) {
                atomicExch(&wordPresence[i * MAX_WORDS + fileId], 1);
            }
        }
    }
}

int main() {
    vector<string> files;
    for (const auto &entry : fs::directory_iterator(".")) {
        if (entry.path().extension() == ".txt") {
            files.push_back(entry.path().string());
        }
    }
    
    unordered_set<string> uniqueWordsSet;
    vector<vector<string>> fileWords(files.size());
    
    for (size_t i = 0; i < files.size(); i++) {
        ifstream file(files[i]);
        string word;
        while (file >> word) {
            uniqueWordsSet.insert(word);
            fileWords[i].push_back(word);
        }
    }
    
    vector<string> uniqueWords(uniqueWordsSet.begin(), uniqueWordsSet.end());
    int wordCount = uniqueWords.size();
    int totalWords = 0;
    for (const auto &fw : fileWords) totalWords += fw.size();

    char *d_words, *d_uniqueWords;
    int *d_fileIds, *d_wordPresence;
    
    cudaMalloc((void**)&d_words, totalWords * MAX_WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_uniqueWords, wordCount * MAX_WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_fileIds, totalWords * sizeof(int));
    cudaMalloc((void**)&d_wordPresence, wordCount * files.size() * sizeof(int));
    
    cudaMemcpy(d_uniqueWords, uniqueWords.data(), wordCount * MAX_WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    
    int offset = 0;
    for (size_t i = 0; i < files.size(); i++) {
        cudaMemcpy(d_words + offset * MAX_WORD_LENGTH, fileWords[i].data(), fileWords[i].size() * MAX_WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
        processWords<<<(fileWords[i].size() + 255) / 256, 256>>>(d_words, d_fileIds, fileWords[i].size(), d_uniqueWords, d_wordPresence, wordCount, i);
        offset += fileWords[i].size();
    }
    
    vector<int> wordPresence(wordCount * files.size());
    cudaMemcpy(wordPresence.data(), d_wordPresence, wordCount * files.size() * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_words);
    cudaFree(d_uniqueWords);
    cudaFree(d_fileIds);
    cudaFree(d_wordPresence);
    
    cout << "\nInverted Index:\n";
    for (size_t i = 0; i < uniqueWords.size(); i++) {
        cout << uniqueWords[i] << " -> ";
        for (size_t j = 0; j < files.size(); j++) {
            if (wordPresence[i * files.size() + j]) {
                cout << files[j] << " ";
            }
        }
        cout << "\n";
    }
    
    return 0;
}
