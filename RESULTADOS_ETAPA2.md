# Resultados da Etapa 2

Comparativo executado em 01/09/2026 com o índice FAISS local de 5.718 chunks. A busca sem filtro usa Dense + BM25 com RRF; a busca filtrada usa o mesmo pipeline, restringindo simultaneamente `doc_type`, `state` e `module`.

| Pergunta | Filtros validados | Primeiro resultado sem filtro | Resultados com filtro |
|---|---|---|---|
| Tickets de Minas Gerais relacionados ao módulo de estoque | `doc_type=ticket`, `state=MG`, `module=estoque` | `md-2026-03-q1_results_exec_review-secao-2-parte-1` | `tickets-TCK-1001`, `tickets-TCK-1004`, `tickets-TCK-1002`, `tickets-TCK-1006` |
| Tickets de São Paulo relacionados ao módulo de PDV | `doc_type=ticket`, `state=SP`, `module=pdv` | `sales-CUST108-481` | `tickets-TCK-1008`, `tickets-TCK-1003` |
| Tickets do Rio de Janeiro relacionados ao módulo pay | `doc_type=ticket`, `state=RJ`, `module=pay` | `sales-CUST1875-1360` | `tickets-TCK-1057` |

## Leitura dos resultados

- Sem filtro, vendas, atas e e-mails podem superar tickets semanticamente relacionados.
- Com filtro, todos os resultados pertencem a `tickets.jsonl` e satisfazem os três metadados extraídos.
- A quantidade inferior a cinco em algumas consultas é esperada: o retriever não completa o ranking com documentos que violem o filtro.
- O resultado demonstra por que o filtro deve ser aplicado antes da fusão final e por que um filtro inválido deve retornar uma lista vazia, nunca uma busca aberta.

Para reproduzir o comparativo completo, com conteúdo e metadados de cada chunk:

```powershell
.\venv\Scripts\python.exe -m src.retrieval_check
```
