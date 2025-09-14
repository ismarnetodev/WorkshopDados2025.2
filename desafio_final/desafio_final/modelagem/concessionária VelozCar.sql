 CREATE DATABASE Concessionaria_VelozCar;
USE Concessionaria_VelozCar;

-- Criação das tabelas

CREATE TABLE veiculo(
    id_veiculo INT PRIMARY KEY AUTO_INCREMENT,
    placa VARCHAR(8) UNIQUE NOT NULL,
    marca VARCHAR(50),
    modelo VARCHAR(50),
    ano_fabricacao YEAR, 
    cor VARCHAR(50),
    tipo_combustivel VARCHAR(50),
    quilometragem DECIMAL(10,2),
    status_veiculo VARCHAR(50)
);

CREATE TABLE cliente(
    id_cliente INT PRIMARY KEY AUTO_INCREMENT,
    nome_cliente VARCHAR(90) NOT NULL,
    endereco VARCHAR(250),
    cpf VARCHAR(11) UNIQUE NOT NULL,
    data_nascimento DATE,
    telefone VARCHAR(15),
    email VARCHAR(100),
    numero_cnh VARCHAR(20),
    validade_cnh DATE
);

CREATE TABLE funcionario(
	id_funcionario INT PRIMARY KEY AUTO_INCREMENT,
	nome_funcionario VARCHAR(90) NOT NULL,
	cpf VARCHAR(11) UNIQUE NOT NULL,
	cargo VARCHAR(50),
	salario DECIMAL(10,2),
	data_contratacao DATE
);	

CREATE TABLE manutencao(
	id_manutencao INT PRIMARY KEY AUTO_INCREMENT,
    id_veiculo INT,
    FOREIGN KEY (id_veiculo) REFERENCES veiculo(id_veiculo),
    data_inicio DATE,
    tipo_manutencao VARCHAR(90),
    descricao VARCHAR(90),
    custo DECIMAL(6,2),
    data_fim DATE
);

CREATE TABLE aluguel(
    id_aluguel INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT, 
    id_veiculo INT,
    data_retirada DATE,
    data_devolucao_prevista DATE,
    data_devolucao_efetiva DATE,
    valor_diarias DECIMAL (10,2),
    valor_seguro DECIMAL (10,2),
    valor_multas DECIMAL(10,2),
    status_aluguel VARCHAR(50),
    FOREIGN KEY (id_cliente) REFERENCES cliente(id_cliente),
    FOREIGN KEY (id_veiculo) REFERENCES veiculo(id_veiculo)
);

CREATE TABLE pagamento(
    id_pagamento INT PRIMARY KEY AUTO_INCREMENT,
    id_aluguel INT,
    FOREIGN KEY (id_aluguel) REFERENCES aluguel(id_aluguel),
    data_pagamento DATE,
    valor_pago DECIMAL(10,2),
    metodo_pagamento VARCHAR(50),
    status_pagamento VARCHAR(50)
);  

-- DADOS

INSERT INTO cliente (nome_cliente, endereco, cpf, data_nascimento, telefone, email, numero_cnh, validade_cnh)
VALUES
('João Silva', 'Rua A, 123', '12345678901', '1990-05-12', '83999990000', 'joao@email.com', '123456', '2027-05-01'),
('Maria Souza', 'Av. Central, 500', '98765432100', '1985-11-30', '83988887777', 'maria@email.com', '654321', '2026-10-15'),
('Pedro Oliveira', 'Rua B, 45', '11122233344', '1992-07-08', '83991112222', 'pedro@email.com', '789456', '2028-01-20'),
('Ana Costa', 'Rua C, 67', '22233344455', '1995-01-15', '83992223333', 'ana@email.com', '987123', '2027-08-05'),
('Lucas Mendes', 'Av. Norte, 999', '33344455566', '1989-03-19', '83993334444', 'lucas@email.com', '456987', '2029-04-30'),
('Fernanda Lima', 'Rua D, 12', '44455566677', '1998-12-25', '83994445555', 'fernanda@email.com', '654987', '2028-06-18'),
('Rafael Souza', 'Rua E, 321', '55566677788', '1983-09-10', '83995556666', 'rafael@email.com', '852147', '2027-12-11'),
('Carla Nunes', 'Av. Sul, 789', '66677788899', '1991-04-22', '83996667777', 'carla@email.com', '963258', '2026-03-01');

INSERT INTO funcionario (nome_funcionario, cpf, cargo, salario, data_contratacao)
VALUES
('Carlos Mendes', '11223344556', 'Atendente', 2500.00, '2022-01-15'),
('Ana Paula', '22334455667', 'Gerente', 4500.00, '2020-03-20'),
('José Almeida', '33445566778', 'Mecânico', 3200.00, '2021-07-01'),
('Mariana Costa', '44556677889', 'Atendente', 2400.00, '2023-02-10'),
('Ricardo Lopes', '55667788990', 'Supervisor', 3800.00, '2019-06-05'),
('Juliana Martins', '66778899001', 'Recepcionista', 2100.00, '2022-11-11'),
('Fernando Rocha', '77889900112', 'Gerente', 4700.00, '2018-04-03'),
('Patrícia Silva', '88990011223', 'Mecânica', 3100.00, '2021-09-09');

INSERT INTO manutencao (id_veiculo, data_inicio, data_fim, tipo_manutencao, descricao, custo)
VALUES
(1, '2023-01-10', '2023-01-12', 'Revisão', 'Troca de óleo e filtros', 450.00),
(2, '2023-02-05', '2023-02-07', 'Freios', 'Troca das pastilhas de freio', 650.00),
(3, '2023-03-15', '2023-03-16', 'Motor', 'Troca de correia dentada', 1200.00),
(4, '2023-04-20', '2023-04-21', 'Pneus', 'Troca de 2 pneus dianteiros', 800.00),
(5, '2023-05-02', '2023-05-03', 'Revisão', 'Revisão geral 20.000km', 600.00),
(6, '2023-06-11', '2023-06-12', 'Suspensão', 'Troca de amortecedores', 1500.00),
(7, '2023-07-07', '2023-07-09', 'Ar-condicionado', 'Carga de gás', 300.00),
(8, '2023-08-01', '2023-08-02', 'Freios', 'Troca de fluído de freio', 250.00);

INSERT INTO veiculo (placa, marca, modelo, ano_fabricacao, cor, tipo_combustivel, quilometragem, status_veiculo)
VALUES
('ABC1234', 'Toyota', 'Corolla', 2020, 'Preto', 'Gasolina', 35000.50, 'Disponível'),
('XYZ5678', 'Honda', 'Civic', 2019, 'Prata', 'Flex', 42000.00, 'Disponível'),
('JKL4321', 'Ford', 'Fiesta', 2018, 'Branco', 'Flex', 58000.20, 'Alugado'),
('MNO8765', 'Chevrolet', 'Onix', 2021, 'Vermelho', 'Gasolina', 15000.00, 'Disponível'),
('PQR3456', 'Hyundai', 'HB20', 2022, 'Azul', 'Flex', 12000.30, 'Disponível'),
('STU6543', 'Volkswagen', 'Gol', 2017, 'Preto', 'Etanol', 69000.90, 'Alugado'),
('VWX7890', 'Fiat', 'Argo', 2019, 'Cinza', 'Flex', 31000.10, 'Disponível'),
('YZA0987', 'Renault', 'Kwid', 2021, 'Branco', 'Gasolina', 8000.00, 'Disponível');

INSERT INTO aluguel (id_cliente, id_veiculo, data_retirada, data_devolucao_prevista, data_devolucao_efetiva, valor_diarias, valor_seguro, valor_multas, status_aluguel)
VALUES
(1, 1, '2023-03-01', '2023-03-07', '2023-03-07', 1500.00, 200.00, 0.00, 'Concluído'),
(2, 2, '2023-04-10', '2023-04-15', NULL, 1200.00, 180.00, 0.00, 'Ativo'),
(3, 3, '2023-05-05', '2023-05-10', '2023-05-11', 1400.00, 190.00, 100.00, 'Concluído'),
(4, 4, '2023-06-01', '2023-06-06', '2023-06-06', 1250.00, 175.00, 0.00, 'Concluído'),
(5, 5, '2023-07-03', '2023-07-08', NULL, 1350.00, 180.00, 0.00, 'Ativo'),
(6, 6, '2023-08-15', '2023-08-20', '2023-08-21', 1450.00, 200.00, 50.00, 'Concluído'),
(7, 7, '2023-09-01', '2023-09-05', NULL, 1100.00, 150.00, 0.00, 'Ativo'),
(8, 8, '2023-09-10', '2023-09-15', NULL, 1000.00, 140.00, 0.00, 'Ativo');

INSERT INTO pagamento (id_aluguel, data_pagamento, valor_pago, metodo_pagamento, status_pagamento)
VALUES
(1, '2023-03-07', 1700.00, 'Cartão de Crédito', 'Pago'),
(2, '2023-04-11', 600.00, 'Pix', 'Parcial'),
(3, '2023-05-11', 1690.00, 'Boleto', 'Pago'),
(4, '2023-06-06', 1425.00, 'Cartão de Débito', 'Pago'),
(5, '2023-07-05', 800.00, 'Pix', 'Parcial'),
(6, '2023-08-21', 1700.00, 'Dinheiro', 'Pago'),
(7, '2023-09-02', 1250.00, 'Cartão de Crédito', 'Pago'),
(8, '2023-09-11', 1140.00, 'Pix', 'Parcial');

SELECT YEAR(data_nascimento) AS ano_nascimento, COUNT(*) AS total_clientes
FROM cliente
GROUP BY YEAR(data_nascimento);

-- Funcionário
SELECT cargo, AVG(salario) AS media_salarial
FROM funcionario
GROUP BY cargo;

-- Veículo
SELECT marca, COUNT(*) AS total_veiculos
FROM veiculo
GROUP BY marca;

-- Manutenção
SELECT tipo_manutencao, SUM(custo) AS custo_total
FROM manutencao
GROUP BY tipo_manutencao;

-- Pagamento
SELECT metodo_pagamento, AVG(valor_pago) AS media_pagamento
FROM pagamento
GROUP BY metodo_pagamento;

-- UPDATE de dados

UPDATE cliente 
SET email = 'pedrowalece@gmail.com',
	telefone = '3526333324'
WHERE id_cliente = 3;

UPDATE cliente 
SET endereco = 'Monte Castelo, 73'
WHERE id_cliente = 4;

-- INNER JOIN entre cliente e aluguel
SELECT c.nome_cliente, a.id_aluguel, a.status_aluguel, a.valor_diarias
FROM cliente c
INNER JOIN aluguel a ON c.id_cliente = a.id_cliente;

-- LEFT JOIN entre veiculo e manutencao
SELECT v.modelo, v.placa, m.tipo_manutencao, m.custo
FROM veiculo v
LEFT JOIN manutencao m ON v.id_veiculo = m.id_veiculo;

SELECT * FROM cliente;
SELECT * FROM funcionario;
SELECT * FROM veiculo;
SELECT * FROM manutencao;
SELECT * FROM aluguel;
SELECT * FROM pagamento;