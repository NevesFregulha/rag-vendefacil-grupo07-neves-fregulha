# Acompanhamento - Mini Desafio RAG VendeFácil

**Integrante 1:** Fernanda Fregulha - [@fregulha](https://github.com/fregulha)
**Integrante 2:** Ester Da Silva Antonio Nóbrega Neves - [@EsterNevess](https://github.com/EsterNevess)

**Repositório:** `rag-vendefacil-grupo07-neves-fregulha`

---

## Como preencher

- Um bloco por encontro, em **ordem cronológica** - o encontro mais recente vai no **fim** do arquivo.
- O relato individual é escrito **pelo próprio integrante**, em primeira pessoa. Não escreva pelo colega.
- Escrever entre **17:30 e 17:40**. `commit` + `push` até as **18:00**, mesmo que o dia não tenha fechado.
- Mensagem de commit: `acompanhamento: AAAA-MM-DD`

**Um relato útil responde:** o que eu implementei, qual decisão técnica eu tomei e por quê, onde travei, e como (ou se) resolvi.

<details>
<summary>Exemplo de relato individual bom × ruim</summary>

❌ *"Trabalhei na parte de ingestão junto com meu colega. Avançamos bastante e conseguimos carregar os arquivos."*

✅ *"Implementei os loaders de CSV e JSONL em `src/ingest.py`. Decidi serializar cada linha do `customers.csv` como frase em linguagem natural em vez de manter o formato separado por vírgula, porque nos primeiros testes de similaridade os chunks CSV crus não recuperavam nada - o embedding não separa campo de valor. Travei ~40 min no `tickets.jsonl`: o `state` estava indo para o texto do chunk mas não para os metadados, então o filtro voltava vazio. Resolvi movendo a extração para antes da criação do `Document`. Usei o Claude para gerar o esqueleto do parser de JSONL; ajustei o schema de metadados na mão."*

</details>

---

## Encontro 1 - 2026-08-24

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - [Nome do Integrante 1]

Fiz, junto com a Ester, a configuração inicial do projeto no Windows. Criamos o ambiente virtual usando Python 3.14.4 e adaptamos os comandos do guia para funcionar no PowerShell, já que alguns estavam escritos para Linux. Depois instalamos as bibliotecas iniciais do RAG, como LangChain, FAISS, Pydantic, Pypdf, BM25 e python-dotenv, e rodamos o `pip check` para confirmar que estava tudo certo. Também exploramos as pastas da base e vimos que teremos arquivos CSV, JSON, JSONL, Markdown, PDF e TXT. Como essa aula ficou focada na preparação, deixamos a criação das pastas e do `src/ingest.py` para a próxima aula. Usei o GitHub Copilot para adaptar os comandos para o Windows e entender os próximos passos.

### Relato individual - [Ester da Silva Antonio Nóbrega Neves]

<!-- Fiz a preparação do projeto junto com a Fernanda. Na minha máquina, acompanhei a criação e a ativação do ambiente virtual no Windows, além da instalação das bibliotecas que vamos usar no desafio. Conferimos o `pip check` e o ambiente ficou funcionando normalmente. Também exploramos a pasta `data` para entender os tipos de arquivo disponíveis e percebemos que cada formato vai precisar de um tratamento diferente na etapa de ingestão. Não começamos o código nesta aula, então combinamos de criar as pastas e iniciar o `src/ingest.py` no próximo encontro. Usamos o GitHub Copilot para entender os comandos e adaptar as instruções para o PowerShell. -->

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Configuramos o ambiente virtual Python e instalamos as dependências iniciais do projeto.
- Validamos o ambiente com `python` e `pip check`.
- Exploramos a organização dos dados e identificamos os formatos que serão processados.

**Ficou pendente:**
- Criar as pastas e arquivos iniciais da aplicação.
- Implementar os leitores e o chunking adaptativo para CSV, JSON, JSONL, Markdown, PDF e TXT.
- Atualizar o arquivo de dependências para registrar as bibliotecas instaladas.

**Bloqueios em aberto:**
- Não tivemos bloqueios durante a configuração do ambiente.

**Próximo passo (início do encontro 2):**
- Criar as pastas iniciais do projeto, atualizar o arquivo de dependências e começar o `src/ingest.py`, iniciando pelos leitores dos arquivos da base.

**Uso de assistentes de IA:**
- Utilizamos o GitHub Copilot para tirar dúvidas, entender os comandos e adaptar as instruções para o PowerShell do Windows. Também conferimos os resultados e validamos a instalação com `pip check`.

---

## Encontro 2 - AAAA-MM-DD

**Etapa:** 2 - Busca híbrida e filtragem por metadados

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Próximo passo (início do encontro 3):**
-

**Uso de assistentes de IA:**
-

---

## Encontro 3 - AAAA-MM-DD

**Etapa:** 3 - Síntese estruturada, evidência e guardrails de LGPD

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Próximo passo (início do encontro 4):**
-

**Uso de assistentes de IA:**
-

---

## Encontro 4 - AAAA-MM-DD

**Etapa:** 4 - Avaliação (RAG Triad), interface e relatório

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Preparação para o Demo Day:**
-

**Uso de assistentes de IA:**
-

---

*TIC em Trilhas · PUC-Rio · Instituto ECOA · MCTI Futuro · Softex*
