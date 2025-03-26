#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <string>
#include <filesystem>
#include <chrono>

using namespace std;
namespace fs = std::filesystem;

#define MAX_WORD_LENGTH 50
#define CHUNK_SIZE (1024 * 1024 * 32)  // 32MB chunks
#define NUM_STREAMS 38

__device__ bool d_isspace(char c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

__device__ char d_tolower(char c) {
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

__device__ int d_strlen(const char *str) {
  int len = 0;
  while (str[len] != '\0') len++;
  return len;
}

__device__ bool d_strncmp(const char *s1, const char *s2, int n) {
  for (int i = 0; i < n; i++) {
    if (s1[i] != s2[i]) return false;
    if (s1[i] == '\0') return true;
  }
  return true;
}

__global__ void processChunkKernel(char *chunk, size_t chunkSize, char *words, int *wordCount, bool isFirstChunk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= chunkSize) return;

  __shared__ char sharedChunk[1024];
  __shared__ bool isWordStart[1024];

  if (threadIdx.x < 1024 && idx < chunkSize) {
      sharedChunk[threadIdx.x] = chunk[idx];
      isWordStart[threadIdx.x] = (idx == 0) || d_isspace(chunk[idx - 1]);
  }
  __syncthreads();

  if (!isFirstChunk && idx == 0) return;

  if (isWordStart[threadIdx.x] && !d_isspace(sharedChunk[threadIdx.x])) {
    char word[MAX_WORD_LENGTH];
    int length = 0;

    while (idx + length < chunkSize && length < MAX_WORD_LENGTH - 1 && !d_isspace(chunk[idx + length])) {
      word[length] = d_tolower(chunk[idx + length]);
      length++;
    }

    if (idx + length >= chunkSize) return;

    word[length] = '\0';

    if (length > 3) {
      const char *suffixes[] = {"ing", "ed", "ly", "ful", "est", "ity", "es", "s"};
      for (const char *suffix : suffixes) {
        int suffixLen = d_strlen(suffix);
        if (length > suffixLen + 1 && 
          d_strncmp(&word[length - suffixLen], suffix, suffixLen)) {
          length -= suffixLen;
          word[length] = '\0';
          break;
        }
      }
    }

    if (length > 0) {
      int wordIdx = atomicAdd(wordCount, 1);
      memcpy(&words[wordIdx * MAX_WORD_LENGTH], word, length + 1);
    }
  }
}

void processFile(const string& filename, unordered_set<string>& uniqueWords) {
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
  }

  ifstream file(filename, ios::binary);
  file.seekg(0, ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, ios::beg);

  char *h_chunk, *d_chunk;
  char *d_words;
  int *d_wordCount;

  cudaMallocHost(&h_chunk, CHUNK_SIZE);
  cudaMalloc(&d_chunk, CHUNK_SIZE);
  cudaMalloc(&d_words, CHUNK_SIZE * MAX_WORD_LENGTH);
  cudaMalloc(&d_wordCount, sizeof(int));

  int currentStream = 0;
  unordered_set<string> fileWords;

  for (size_t offset = 0; offset < fileSize; offset += CHUNK_SIZE) {
  size_t currentChunkSize = std::min(static_cast<size_t>(CHUNK_SIZE), fileSize - offset);
  file.read(h_chunk, currentChunkSize);

  cudaMemsetAsync(d_wordCount, 0, sizeof(int), streams[currentStream]);
  cudaMemcpyAsync(d_chunk, h_chunk, currentChunkSize, cudaMemcpyHostToDevice, streams[currentStream]);

    int blockSize = 256;
    int numBlocks = (currentChunkSize + blockSize - 1) / blockSize;

    processChunkKernel<<<numBlocks, blockSize, 0, streams[currentStream]>>>(d_chunk, currentChunkSize, d_words, d_wordCount, offset == 0);

    int h_wordCount;
    cudaMemcpyAsync(&h_wordCount, d_wordCount, sizeof(int), cudaMemcpyDeviceToHost, streams[currentStream]);

    cudaStreamSynchronize(streams[currentStream]);

    if (h_wordCount > 0) {
      char *h_words = new char[h_wordCount * MAX_WORD_LENGTH];
      cudaMemcpy(h_words, d_words, h_wordCount * MAX_WORD_LENGTH, cudaMemcpyDeviceToHost);

      for (int i = 0; i < h_wordCount; i++) {
        string word(&h_words[i * MAX_WORD_LENGTH]);
        if (fileWords.insert(word).second) {
          uniqueWords.insert(word);
        }
      }

      delete[] h_words;
    }

    currentStream = (currentStream + 1) % NUM_STREAMS;
    float progress = (float)offset / fileSize * 100;
    cout << "\rProcessing " << fs::path(filename).filename().string() << ": " << progress << "%" << flush;
  }
  cout << endl;

  cudaFreeHost(h_chunk);
  cudaFree(d_chunk);
  cudaFree(d_words);
  cudaFree(d_wordCount);

  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
}

int main() {
  auto start_time = chrono::high_resolution_clock::now();

  vector<string> files;
  for (const auto &entry : fs::directory_iterator("GeneratedFiles")) {
    if (entry.path().extension() == ".txt") {
      files.push_back(entry.path().string());
    }
  }

  if (files.empty()) {
    cout << "No .txt files found in GeneratedFiles directory!" << endl;
    return 1;
  }

  unordered_set<string> globalUniqueWords;
  vector<unordered_set<string>> fileUniqueWords(files.size());

  for (size_t i = 0; i < files.size(); i++) {
    processFile(files[i], fileUniqueWords[i]);
    globalUniqueWords.insert(fileUniqueWords[i].begin(), fileUniqueWords[i].end());
  }

  cout << "\nGenerating Inverted Index...\n";
  for (const auto& word : globalUniqueWords) {
    cout << word << " -> ";
    for (size_t i = 0; i < files.size(); i++) {
      if (fileUniqueWords[i].find(word) != fileUniqueWords[i].end()) {
        cout << fs::path(files[i]).filename().string() << " ";
      }
    }
    cout << "\n";
  }

  auto end_time = chrono::high_resolution_clock::now();
  float total_time = chrono::duration<float>(end_time - start_time).count();
  cout << "\nTotal processing time: " << total_time << " seconds\n";

  return 0;
}

// Compile: nvcc -O3 -arch=sm_89 -std=c++17 InvertedIndexCU.cu -o InvertedIndexCU.exe
// Run:     InvertedIndexCU.exe