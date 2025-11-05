#include "postgres.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>

// Function to read binary embeddings file
bool read_embeddings(const std::string& filepath, 
                     std::vector<std::vector<float>>& embeddings, 
                     size_t embedding_dim) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open embeddings file: " << filepath << std::endl;
        return false;
    }

    // Read number of embeddings
    int32_t num_embeddings;
    file.read(reinterpret_cast<char*>(&num_embeddings), sizeof(int32_t));
    
    // Read embedding dimension
    int32_t dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    
    if (dim != static_cast<int32_t>(embedding_dim)) {
        std::cerr << "❌ Embedding dimension mismatch. Expected: " << embedding_dim 
                  << ", Got: " << dim << std::endl;
        return false;
    }

    std::cout << "📊 Reading " << num_embeddings << " embeddings of dimension " << dim << std::endl;

    // Read each embedding
    embeddings.reserve(num_embeddings);
    for (int32_t i = 0; i < num_embeddings; ++i) {
        std::vector<float> embedding(dim);
        file.read(reinterpret_cast<char*>(embedding.data()), dim * sizeof(float));
        embeddings.push_back(std::move(embedding));
    }

    file.close();
    std::cout << "✅ Successfully loaded " << embeddings.size() << " embeddings" << std::endl;
    return true;
}

// Function to read usernames file
bool read_usernames(const std::string& filepath, std::vector<std::string>& usernames) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open usernames file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing whitespace/newlines
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        if (!line.empty()) {
            usernames.push_back(line);
        }
    }

    file.close();
    std::cout << "✅ Successfully loaded " << usernames.size() << " usernames" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    // Default paths
    std::string data_path = "../../data";
    std::string embeddings_file = data_path + "/embeddings.bin";
    std::string usernames_file = data_path + "/usernames.txt";
    size_t embedding_dim = 512;

    // Database configuration (can be passed as environment variables)
    std::string db_host = "localhost";
    int db_port = 5433;
    std::string db_name = "healthmed";
    std::string db_user = "paperless";
    std::string db_password = "paperless";

    // Parse command line arguments
    if (argc > 1) {
        embeddings_file = argv[1];
    }
    if (argc > 2) {
        usernames_file = argv[2];
    }
    if (argc > 3) {
        embedding_dim = std::stoul(argv[3]);
    }

    std::cout << "🚀 Starting data migration to PostgreSQL..." << std::endl;
    std::cout << "📁 Embeddings file: " << embeddings_file << std::endl;
    std::cout << "📁 Usernames file: " << usernames_file << std::endl;
    std::cout << "🔢 Embedding dimension: " << embedding_dim << std::endl;

    // Load data from files
    std::vector<std::vector<float>> embeddings;
    std::vector<std::string> usernames;

    if (!read_embeddings(embeddings_file, embeddings, embedding_dim)) {
        std::cerr << "❌ Failed to read embeddings. Make sure to export data first." << std::endl;
        std::cerr << "💡 Run: python3 export_data.py" << std::endl;
        return 1;
    }

    if (!read_usernames(usernames_file, usernames)) {
        std::cerr << "❌ Failed to read usernames" << std::endl;
        return 1;
    }

    // Validate data consistency
    if (embeddings.size() != usernames.size()) {
        std::cerr << "❌ Data mismatch: " << embeddings.size() << " embeddings but " 
                  << usernames.size() << " usernames" << std::endl;
        return 1;
    }

    std::cout << "✅ Data validation passed" << std::endl;

    try {
        // Connect to PostgreSQL
        std::cout << "🔌 Connecting to PostgreSQL..." << std::endl;
        Postgres db(db_host, db_port, db_name, db_user, db_password);
        std::cout << "✅ Connected to database" << std::endl;

        // Insert data
        std::cout << "📝 Inserting data into database..." << std::endl;
        size_t success_count = 0;
        size_t error_count = 0;

        for (size_t i = 0; i < embeddings.size(); ++i) {
            try {
                db.insert_embedding(usernames[i], embeddings[i]);
                success_count++;
                
                if ((i + 1) % 10 == 0 || (i + 1) == embeddings.size()) {
                    std::cout << "⏳ Progress: " << (i + 1) << "/" << embeddings.size() 
                              << " (" << (100 * (i + 1) / embeddings.size()) << "%)" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Failed to insert " << usernames[i] << ": " << e.what() << std::endl;
                error_count++;
            }
        }

        std::cout << "\n✅ Migration completed!" << std::endl;
        std::cout << "   Successfully inserted: " << success_count << std::endl;
        std::cout << "   Errors: " << error_count << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Database error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

