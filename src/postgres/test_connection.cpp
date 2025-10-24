#include "postgres.hpp"
#include <iostream>
#include <sstream>

int main() {
    try {
        pqxx::connection conn(
            "host=localhost port=5433 dbname=healthmed user=paperless password=paperless"
        );

        if (!conn.is_open()) {
            std::cerr << "DB failed to connect" << std::endl;
            return 1;
        }
        std::cout << "DB connected successfully" << std::endl;

        // pqxx::work txn(conn);
        // pqxx::result r = txn.exec("CREATE TABLE Person(id BIGSERIAL PRIMARY KEY, name VARCHAR(255), embedding VECTOR(512))");
        // std::cout << "Table created successfully" << std::endl;
        // txn.commit();

        pqxx::work txnn(conn);
        pqxx::result r = txnn.exec_params("SELECT * FROM Person");
        std::cout << "Result: " << r.size() << std::endl;
        for (const auto& row : r) {
            std::cout << "Name: " << row["name"].as<std::string>() << std::endl;
            std::cout << "Embedding: " << row["embedding"].size() << std::endl;
            std::vector<float> embedding = pgvector2vec(row["embedding"].as<std::string>());
            std::cout << "Embedding size: " << embedding.size() << std::endl;
        }
        txnn.commit();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
