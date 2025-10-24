#include "postgres.hpp"

// DBPerson implementation
DBPerson::DBPerson(
    const uint64_t& id,
    const std::string& name,
    const float& confidence,
    const float& distance
) : id(id), name(name), confidence(confidence), distance(distance) {
}

std::string DBPerson::to_string() const {
    return "ID: " + std::to_string(id) + 
           ", Name: " + name + 
           ", Confidence: " + std::to_string(confidence) + 
           ", Distance: " + std::to_string(distance);
}

// Postgres implementation
Postgres::Postgres(
    const std::string& host,
    const int& port,
    const std::string& dbname,
    const std::string& user,
    const std::string& password
) : conn("host=" + host + 
         " port=" + std::to_string(port) + 
         " dbname=" + dbname + 
         " user=" + user + 
         " password=" + password) {
}

Postgres::~Postgres() {
    conn.disconnect();
}

void Postgres::insert_embedding(const std::string& name, const std::vector<float>& embedding) {
    pqxx::work txn(conn);
    txn.exec_params(
        "INSERT INTO Person(name, embedding) VALUES ($1, $2)",
        name,
        vec2pgvector(embedding)
    );
    txn.commit();
}

void Postgres::update_embedding(const uint64_t& id, const std::string& name, const std::vector<float>& embedding) {
    pqxx::work txn(conn);
    txn.exec_params(
        "UPDATE Person SET name = $1, embedding = $2 WHERE id = $3",
        name,
        vec2pgvector(embedding),
        id
    );
    txn.commit();
}

std::vector<float> Postgres::get_embedding(const std::string& name) {
    pqxx::work txn(conn);
    pqxx::result r = txn.exec_params(
        "SELECT embedding FROM Person WHERE name = $1",
        name
    );
    txn.commit();
    return pgvector2vec(r[0]["embedding"].as<std::string>());
}

std::vector<DBPerson> Postgres::get_persons(std::string name) {
    pqxx::work txn(conn);
    pqxx::result r = txn.exec_params(
        "SELECT id, name FROM Person WHERE name = $1",
        name
    );
    txn.commit();
    std::vector<DBPerson> persons;
    for (const auto& row : r) {
        persons.push_back(DBPerson(
            row["id"].as<uint64_t>(), 
            row["name"].as<std::string>(), 
            0.0, 
            0.0
        ));
    }
    return persons;
}

DBPerson Postgres::get_recognition(const std::vector<float>& embedding, float threshold) {
    pqxx::work txn(conn);
    pqxx::result r = txn.exec_params(
        "SELECT id, name, distance "
        "FROM ( "
        "   SELECT id, name, embedding <=> $1::vector AS distance "
        "   FROM Person "
        ") AS sub "
        "WHERE distance < $2 "
        "ORDER BY distance "
        "LIMIT 1;",
        vec2pgvector(embedding),
        threshold
    );
    txn.commit();
    
    if (r.empty()) {
        return DBPerson(0, "", 0.0, 0.0);
    }
    
    return DBPerson(
        r[0]["id"].as<uint64_t>(),
        r[0]["name"].as<std::string>(),
        1 - r[0]["distance"].as<float>(),
        r[0]["distance"].as<float>()
    );
}

