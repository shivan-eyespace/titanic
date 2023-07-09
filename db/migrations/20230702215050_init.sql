-- migrate:up
CREATE TYPE sex_type AS ENUM ('m', 'f');
CREATE TYPE pclass_type AS ENUM ('1', '2', '3');
CREATE TYPE embarked_type AS ENUM ('S', 'C', 'Q');
CREATE TABLE passengers (
    id SERIAL PRIMARY KEY,
    pclass PCLASS_TYPE NOT NULL,
    sex SEX_TYPE NOT NULL,
    age INTEGER NOT NULL,
    sibsp INTEGER NOT NULL,
    parch INTEGER NOT NULL,
    fare REAL NOT NULL,
    embarked EMBARKED_TYPE NOT NULL
);

-- migrate:down
DROP TABLE passengers;
DROP TYPE SEX_TYPE;
DROP TYPE PCLASS_TYPE;
DROP TYPE EMBARKED_TYPE;
