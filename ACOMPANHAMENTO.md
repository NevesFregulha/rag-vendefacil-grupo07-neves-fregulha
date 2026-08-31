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

### Relato individual - Fernanda Fregulha

Fiz, junto com a Ester, a configuração inicial do projeto no Windows. Criamos o ambiente virtual usando Python 3.14.4 e adaptamos os comandos do guia para funcionar no PowerShell, já que alguns estavam escritos para Linux. Depois instalamos as bibliotecas iniciais do RAG, como LangChain, FAISS, Pydantic, Pypdf, BM25 e python-dotenv, e rodamos o `pip check` para confirmar que estava tudo certo. Também exploramos as pastas da base e vimos que teremos arquivos CSV, JSON, JSONL, Markdown, PDF e TXT. Como essa aula ficou focada na preparação, deixamos a criação das pastas e do `src/ingest.py` para a próxima aula. Usei o GitHub Copilot para adaptar os comandos para o Windows e entender os próximos passos.

### Relato individual - Ester da Silva Antonio Nóbrega Neves

Fiz a preparação do projeto junto com a Fernanda. Na minha máquina, acompanhei a criação e a ativação do ambiente virtual no Windows, além da instalação das bibliotecas que vamos usar no desafio. Conferimos o `pip check` e o ambiente ficou funcionando normalmente. Também exploramos a pasta `data` para entender os tipos de arquivo disponíveis e percebemos que cada formato vai precisar de um tratamento diferente na etapa de ingestão. Não começamos o código nesta aula, então combinamos de criar as pastas e iniciar o `src/ingest.py` no próximo encontro. Usamos o GitHub Copilot para entender os comandos e adaptar as instruções para o PowerShell.

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

## Encontro 2 - 2026-08-26

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - Fernanda Fregulha

Preparei a estrutura compartilhada do projeto em `src/`, `src/loaders/` e `tests/` e implementei em `src/metadata.py` o contrato comum dos metadados, com validação de `source_file`, `doc_type`, `chunk_id` e `sensitivity`, incluindo detecção de IDs duplicados. Depois implementei os loaders de JSONL, Markdown, PDF e TXT com chunking adaptativo: um ticket por chunk no JSONL, divisão por seção nos Markdown, divisão por página e parágrafo nos PDFs e separação de mensagens nos e-mails. Validei 203 chunks textuais e classifiquei como `restrito` cinco e-mails internos que continham credenciais. Criei testes automatizados para os metadados e loaders; os oito testes executados passaram. Também revisei a ingestão estruturada depois do merge e identifiquei que `products.json` e `stores.json` foram tratados como um objeto único e que `system_logs.csv` ainda não foi incluído, deixando 503 registros pendentes. Usei o Codex para gerar esqueletos, revisar os formatos reais da base e executar os testes; conferi os resultados, ajustei as estratégias de chunking e preservei loaders diferentes para cada natureza de documento.

### Relato individual - Ester da Silva Antonio Nóbrega Neves

Acompanhei a preparação inicial do projeto e, nesta TASK 02, implementei a ingestão dos dados estruturados (CSV e JSON) da Etapa 1. Na minha máquina, criei a branch ester/loaders-estruturados e desenvolvi as funções em src/loaders/structured.py para ler os três arquivos CSV e os dois arquivos JSON (customers.csv, employees.csv, sales.csv, products.json e stores.json).
Segui rigorosamente a regra de que cada registro equivale a um Document e um chunk, transformando as linhas e objetos em textos compreensíveis para o modelo de embeddings. Também apliquei o contrato de metadados obrigatórios (source_file, doc_type, chunk_id e sensitivity), gerando IDs únicos e estáveis.
Para validar o código, criei os testes unitários em tests/test_structured.py, executando o pytest e obtendo 100% de aprovação. Por fim, realizei os commits separadamente, publiquei a branch no GitHub e abri o Pull Request (#3) para a main sem utilizar Squash and merge, preservando os commits individuais.

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Estrutura inicial de `src/`, `src/loaders/` e `tests/`, com contrato e validação de metadados.
- Loaders de JSONL, Markdown, PDF e TXT com chunking adaptativo, totalizando 203 chunks textuais validados.
- Loader inicial de CSV e JSON, com 5.012 chunks estruturados atualmente carregados.
- Testes automatizados de metadados, loaders textuais e ingestão estruturada.
- Pull Requests das TASKS 01, 02 e 03 integrados à `main`, preservando os commits individuais.

**Ficou pendente:**
- Corrigir a leitura das listas internas de `products.json` e `stores.json`, que devem gerar 5 e 50 chunks, respectivamente.
- Incluir os 450 registros de `data/semi_structured/system_logs.csv` com `doc_type="log"`.
- Integrar todos os loaders em `src/ingest.py`, criar e persistir o índice FAISS e executar o script de sanidade.
- Atualizar o `README.md` com as instruções e decisões da Etapa 1.

**Bloqueios em aberto:**
- A ingestão estruturada está 503 chunks abaixo do total esperado. Decidimos registrar a pendência e corrigi-la antes de gerar o índice FAISS.

**Próximo passo:**
- Corrigir a ingestão de produtos, lojas e logs; em seguida, executar a TASK 04 para integrar os loaders, gerar o FAISS e validar as três perguntas de sanidade.

**Uso de assistentes de IA:**
- Utilizamos o Codex para orientar o fluxo de branches e Pull Requests, propor os esqueletos dos loaders, revisar os formatos reais dos arquivos e automatizar testes. As sugestões foram conferidas contra o corpus, e ajustamos manualmente as estratégias de chunking, os metadados e a classificação de sensibilidade. A revisão também permitiu identificar os 503 registros ainda ausentes da ingestão estruturada.

---

## Encontro 3 - 2026-08-28

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - Ester da Silva Antonio Nóbrega Neves

Assumi a TASK 04 neste encontro porque a Fernanda não pôde comparecer. Corrigi a ingestão de `products.json` e `stores.json` em `src/loaders/structured.py`, ajustando a leitura das listas internas para gerar um `Document` por registro. Validei que a ingestão passou a gerar 5 documentos de produto e 50 documentos de loja.
Também criei `src/loaders/log_loader.py` para processar os 450 registros de `data/semi_structured/system_logs.csv`, com `doc_type="log"` e metadados filtráveis como `timestamp`, `level`, `service`, `module`, `customer_id`, `event` e `error_code`. Criei testes específicos para os logs e validei os 450 documentos gerados.
Em seguida, integrei todos os loaders em `src/ingest.py`, incluindo dados estruturados, logs, tickets JSONL, Markdown, PDF e e-mails TXT. A ingestão completa totalizou 5.718 chunks e os metadados foram validados quanto aos campos obrigatórios e à unicidade dos `chunk_id`.
Implementei em `src/vectorstore.py` a criação, persistência e recarga do índice FAISS usando embeddings locais multilíngues com o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. O índice é salvo localmente e pode ser recarregado sem reprocessar o corpus. Também implementei `src/sanity_check.py`, que mostra o total de chunks, a distribuição por tipo de documento e os cinco resultados mais similares para três perguntas de teste.
Por fim, atualizei o `README.md` para documentar a execução real da Etapa 1, incluindo instalação, estratégias de chunking, metadados, criação do índice, recarga e sanidade. Executei a suíte completa de testes com resultado de 18 testes aprovados. Usei o Codex para orientar a correção dos loaders, estruturar os testes, revisar os comandos e comparar a implementação com o guia do professor; conferi e executei localmente cada alteração e resultado antes de registrar no repositório.

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Correção da ingestão de produtos, lojas e logs.
- Integração de todos os loaders em `src/ingest.py`.
- Validação do corpus completo com 5.718 chunks.
- Criação, persistência e recarga do índice FAISS local.
- Script de sanidade com distribuição por `doc_type` e três perguntas de teste.
- Testes de logs, ingestão e FAISS.
- Atualização do `README.md` com a documentação da Etapa 1.
- Suíte completa validada com 18 testes aprovados.
- TASK 04 e TASK 05 integradas à `main` por Pull Requests, sem squash.

**Ficou pendente:**
- Iniciar a Etapa 2: analisador de perguntas, filtros por metadados, busca BM25 e busca híbrida.

**Bloqueios em aberto:**
- Não houve bloqueio técnico na conclusão da Etapa 1.
- Durante a leitura dos PDFs, o `pypdf` exibiu avisos sobre referências internas (`startxref`), mas os documentos foram processados normalmente.
- O `langchain-community` exibiu aviso de descontinuação futura, sem impactar a execução atual.

**Próximo passo (início do encontro 4):**
- Criar a branch da Etapa 2 e iniciar a TASK 01: analisador da pergunta e extração de filtros por `state`, `module`, `customer_id` e `priority`.

**Uso de assistentes de IA:**
- Utilizei o Codex para revisar a estrutura do projeto, identificar a causa da ingestão incorreta de produtos e lojas, orientar a criação do loader de logs, estruturar a integração com FAISS, elaborar os testes e revisar a documentação. As sugestões foram conferidas manualmente contra os arquivos do corpus, a saída dos scripts e os resultados da suíte de testes.

---

## Encontro 4 - 2026-08-31

**Etapa:** 2 - Busca híbrida e filtragem por metadados

### Relato individual - Fernanda Fregulha

Neste encontro, revisei as tasks concluídas na Etapa 1 para retomar o contexto da implementação e compreender como os loaders, os metadados e o índice FAISS foram estruturados. Também revisei a organização das tasks da Etapa 2 e usei o dia para me alinhar com a Ester sobre o que já havia sido desenvolvido no analisador de perguntas. Em conjunto, planejamos os próximos passos da etapa, considerando as dependências entre a extração de filtros, o retriever FAISS, o índice BM25, a busca híbrida e os testes de integração. Usei assistentes de IA para revisar alguns conteúdos da atividade e analisar possibilidades de divisão e sequenciamento das próximas tarefas.

### Relato individual - Ester da Silva Antonio Nóbrega Neves

Iniciei a Etapa 2 neste encontro com a TASK 01: o analisador de perguntas e a extração de filtros. Criei a branch ester/analisador-perguntas e implementei em src/query_analyzer.py a função extract_filters(), que reconhece os quatro filtros de metadados a partir de uma pergunta em texto livre: state, module, customer_id e priority. Antes de escrever o código, explorei os valores reais gravados nos metadados pelos loaders da Etapa 1 (customers.csv, stores.json, tickets.jsonl e system_logs.csv) para não inventar formatos que não existem no corpus: os estados aparecem como sigla (ex: MG, SP), os módulos como pdv, estoque, ecommerce, analytics e pay, e a prioridade dos tickets como Alta, Média, Baixa ou Crítica. Para o estado, o analisador reconhece tanto a sigla quanto o nome completo (ex: "Minas Gerais" vira MG); para o módulo, mapeei sinônimos que uma pergunta real usaria, como "frente de caixa" e "ponto de venda" para pdv. 
Resolvi não mapear "loja" isolada e só reconhecer combinações inequívocas como "loja online" e "e-commerce". Também tratei prioridade e nomes de estado com e sem acento (ex: "critica"/"crítica"), e o customer_id com um regex que aceita o padrão CUSTxxx em qualquer capitalização. Criei 11 testes em tests/test_query_analyzer.py, cobrindo cada filtro isolado, a combinação de vários filtros na mesma pergunta e o caso sem nenhum filtro reconhecido. Executei a suíte completa e obtive 29 testes aprovados (18 da Etapa 1 mais os 11 novos), sem quebrar nada da ingestão ou do índice FAISS já existentes. 

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Concluímos a TASK 01 da Etapa 2: analisador da pergunta e extração dos filtros `state`, `module`, `customer_id` e `priority`.
- Revisamos os valores reais dos metadados, definimos regras de normalização e sinônimos e validamos o analisador com testes automatizados.
- Revisamos as entregas anteriores e realizamos ajustes e alinhamentos sobre a integração das próximas partes da Etapa 2.
- Planejamos a sequência de implementação do retriever FAISS com filtros, do índice BM25, da busca híbrida e dos testes de sanidade.

**Ficou pendente:**
- Concluir as demais tasks da Etapa 2: retriever FAISS com filtros por metadados, índice BM25, combinação com a busca vetorial, integração, testes, consultas de sanidade, documentação e relatos individuais.

**Bloqueios em aberto:**
- Não tivemos bloqueios neste encontro.

**Preparação para o Demo Day:**
- A preparação ainda está em fase de construção, acompanhando a evolução das próximas etapas do projeto.

**Uso de assistentes de IA:**
- Utilizamos assistentes de IA para revisar alguns conteúdos da atividade, analisar possibilidades de implementação e apoiar o planejamento das próximas tasks. As sugestões foram discutidas e conferidas pela dupla antes de serem incorporadas ao trabalho.

---

*TIC em Trilhas · PUC-Rio · Instituto ECOA · MCTI Futuro · Softex*
