DROP TABLE IF EXISTS DESAFIO;
DROP TABLE IF EXISTS MARCA;

CREATE TABLE DESAFIO (
    id INT AUTO_INCREMENT PRIMARY KEY,
    carro VARCHAR(100),
    id_marca INT,
    cor VARCHAR(50),
    ano INT
);

CREATE TABLE MARCA (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(50)
);

INSERT INTO MARCA (nome)
VALUES
    ('hyundai'),
    ('ferrari'),
    ('volks'),
    ('fiat'),
    ('honda'),
    ('toyota');

INSERT INTO DESAFIO (carro, id_marca, cor, ano)
VALUES
    ('creta', 1, 'branco', 2021),
    ('sf90', 2, 'vermelha', 2024),
    ('gol', 3, 'branco', 2015),
    ('polo', 3, 'preto', 2019),
    ('mobi', 4, 'azul', 2025),
    ('creta', 1, 'preto', 2025),
    ('saveiro', 3, 'vermelho', 2019),
    ('hrv', 5, 'preto', 2020),
    ('wrv', 5, 'branco', 2024),
    ('hb20', 1, 'vermelho', 2018);

-- DML
UPDATE DESAFIO
SET id_marca = 6
WHERE carro = 'sf90';

-- 3 Funções Agregadas e 2 de Agrupamento
SELECT
    m.nome AS marca,
    COUNT(d.carro) AS total_carros,
    COUNT(DISTINCT d.cor) AS total_cores_diferentes,
    AVG(d.ano) AS ano_medio
FROM
    DESAFIO d
JOIN
    MARCA m ON d.id_marca = m.id
GROUP BY
    m.nome
ORDER BY
    total_carros DESC;

-- 1 JOIN e 1 DQL
SELECT
    d.carro,
    m.nome AS marca,
    d.cor,
    d.ano
FROM
    DESAFIO d
JOIN
    MARCA m ON d.id_marca = m.id;