#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

const long long GB = 1024LL * 1024 * 1024; // 1 GB
const int CHUNK_SIZE = 1024 * 1024;
const int DEFAULT_VOCAB_SIZE = 1000;
const int DEFAULT_FILE_COUNT = 1;
const int DEFAULT_WORDS_TO_REMOVE = 0;

vector<string> load_dictionary(int vocab_size) {
    vector<string> dictionary;
    vector<string> file_paths;
    string folder = "Words";
    
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path().string());
        }
    }

    if (file_paths.empty()) return dictionary;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<size_t> file_dist(0, file_paths.size() - 1);

    for (int i = 0; i < vocab_size; i += 5) {
        string selected_file = file_paths[file_dist(gen)];
        ifstream file(selected_file);
        
        if (!file) continue;

        file.seekg(0, ios::beg);
        size_t line_count = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n');
        
        if (line_count <= 4) continue;

        uniform_int_distribution<size_t> line_dist(2, line_count - 3);
        size_t target_line = line_dist(gen);

        file.seekg(0, ios::beg);
        string line;
        size_t current_line = 0;
        vector<string> lines;

        while (getline(file, line)) {
            if (current_line >= target_line - 2 && current_line <= target_line + 2) {
                line.erase(line.find_last_not_of(" \n\r\t") + 1);
                if (!line.empty()) {
                    lines.push_back(line);
                }
            } else if (current_line > target_line + 2) {
                break;
            }
            current_line++;
        }

        if (lines.size() == 5) {
            dictionary.insert(dictionary.end(), lines.begin(), lines.end());
        }
    }

    shuffle(dictionary.begin(), dictionary.end(), gen);
    return dictionary;
}

vector<string> modify_dictionary(const vector<string>& original_dict, int words_to_remove) {
    vector<string> modified_dict = original_dict;
    random_device rd;
    mt19937 gen(rd());
    
    if (words_to_remove > 0 && words_to_remove < modified_dict.size()) {
        shuffle(modified_dict.begin(), modified_dict.end(), gen);
        modified_dict.erase(modified_dict.begin(), modified_dict.begin() + words_to_remove);
    }

    return modified_dict;
}

int main(int argc, char* argv[]) {
    int vocab_size = DEFAULT_VOCAB_SIZE;
    long long total_size = 20LL * GB;
    int file_count = DEFAULT_FILE_COUNT;
    int words_to_remove = DEFAULT_WORDS_TO_REMOVE;

    if (argc > 1) {
        vocab_size = atoi(argv[1]) * 5;
        if (vocab_size <= 0) {
            cerr << "Error: Vocabulary size must be positive\n";
            return 1;
        }
    }
    if (argc > 2) {
        total_size = atoll(argv[2]) * GB;
        if (total_size <= 0) {
            cerr << "Error: Total file size must be positive\n";
            return 1;
        }
    }
    if (argc > 3) {
        file_count = atoi(argv[3]);
        if (file_count <= 0) {
            cerr << "Error: File count must be positive\n";
            return 1;
        }
    }
    if (argc > 4) {
        words_to_remove = atoi(argv[4]);
        if (words_to_remove < 0) {
            cerr << "Error: Words to remove must be non-negative\n";
            return 1;
        }
    }

    long long file_size = total_size / file_count;
    vector<string> base_dictionary = load_dictionary(vocab_size);
    if (base_dictionary.empty()) {
        cerr << "Error: No words loaded from dictionary\n";
        return 1;
    }

    string output_folder = "GeneratedFiles";
    fs::create_directories(output_folder);

    for (int file_idx = 1; file_idx <= file_count; ++file_idx) {
        vector<string> dictionary = modify_dictionary(base_dictionary, words_to_remove);

        cout << "\nDictionary for file " << file_idx << ":\n";
        for (const auto& word : dictionary) {
            cout << word << " ";
        }
        cout << "\n\n";

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> word_dist(0, dictionary.size() - 1);

        string filename = output_folder + "/large_text_" + to_string(file_idx) + ".txt";
        ofstream file(filename);
        if (!file) {
            cerr << "Error creating file: " << filename << "\n";
            return 1;
        }

        cout << "Generating: " << filename << " (Size: " << file_size / GB << "GB)\n";

        long long written = 0;
        string chunk;
        chunk.reserve(CHUNK_SIZE);
        int lastPercentage = -1;
        const int barWidth = 50;

        while (written < file_size) {
            for (int i = 0; i < 500; ++i) {
                chunk += dictionary[word_dist(gen)] + " ";
            }
            file << chunk;
            written += chunk.size();

            int currentPercentage = (written * 100) / file_size;
            if (currentPercentage != lastPercentage) {
                float progress = (float)written / file_size;
                int pos = barWidth * progress;
                cout << "\r[";
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) cout << "=";
                    else if (i == pos) cout << ">";
                    else cout << " ";
                }
                cout << "] " << currentPercentage << "%" << flush;
                lastPercentage = currentPercentage;
            }

            chunk.clear();
        }

        file.close();
        cout << "\nFile generated: " << filename << "\n";
    }
    return 0;
}


//Compile:    cl /std:c++17 MultipleFilesGenerator.cpp
//Run:        MultipleFilesGenerator.exe 8 20 3 10
//Check File: [System.Text.Encoding]::UTF8.GetString((Get-Content GeneratedFiles/large_text_1.txt -Encoding Byte -TotalCount 1000))