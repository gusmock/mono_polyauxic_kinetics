# Manutenção do Streamlit — 12/11/2026

Build: `streamlit-20261112.1`.
Escopo: aplicação pública `gusmock/mono_polyauxic_kinetics`, derivada do commit
`2bc20a9f5e3e1b75b3ed862cbbccdba414772950`. O código da suíte institucional e os
arquivos históricos de reprodução não integram esta refatoração.

## Solicitação da informática

Chamado **TN-2026111186000044**, prazo **13/11/2026**. O relatório aponta
DOMPurify 3.2.6 e Moment 2.30.1. Mínimos solicitados: 3.4.17 e 2.31.0,
respectivamente. A fila pública de issues estava sem chamados abertos na consulta.

O `requirements.txt` fixa Streamlit **1.67.0** e suas dependências Python diretas
e transitivas. Removidos `streamlit[auth]` e Authlib. Python **≥3.12** é necessário
para NumPy 2.5.2 e SciPy 1.18.0; o ambiente de validação usa Python 3.12.
Não instalar estes requisitos no ambiente de outra aplicação.

Foram inspecionados o lockfile oficial e os arquivos JavaScript do wheel instalado:

| Biblioteca | Versão verificada | Arquivo distribuído |
| --- | --- | --- |
| DOMPurify | 3.4.17 | `purify.es.BXhPiKO5.js` |
| Moment | 2.31.0 | `moment.4ragD6OA.js` |

Wheel Streamlit: `streamlit-1.67.0-py3-none-any.whl`.
SHA-256: `f29da87c443bfbdfc52738616e3d65745d438e718da8faf69e2be0da1e52ce64`.
O verificador `scripts/check_legacy_frontend.py` falha diante de versões antigas,
ausentes ou implementações reconhecidas cuja versão não possa ser determinada.
A verificação é específica dessas bibliotecas; não substitui um scanner geral.

## Correções

- Removidos login, cadastro, coleta de perfil e gravação automática de uploads.
  Arquivos antigos existentes no servidor não são apagados.
- Separados entrada, núcleo matemático, orquestração, tradução, figuras,
  exportação e interface. O legado funciona sem importar a suíte privada.
- Resultados conservados apenas na sessão: idioma, fonte, eixos e downloads
  não repetem o cálculo. Dados, configurações científicas ou interpretação do
  arquivo alterados invalidam o resultado anterior.
- Leitura limitada de CSV/XLSX, separador decimal explícito, rejeição de erros
  em pares, infinitos, fórmulas, macros, links externos e descompactação excessiva.
  Nenhuma linha parcialmente inválida é descartada silenciosamente.
- Ordenação estável mantém a identidade das observações com tempos repetidos
  entre ajuste, máscara de exclusão, gráfico e exportação. Tempos distintos
  não são mais agrupados após arredondamento a quatro casas.
- Corrigida a amplitude da primeira ordem: `k = r_max/(p*abs(y_f-y_i))`.
  A derivada inicial passa a corresponder a `r_max` na resposta crescente.
- Removida a redundância translacional do softmax: último logit fixo em zero.
  Assíntota inicial fixa e lag não utilizado da primeira ordem são excluídos da
  otimização. Contagem explícita de parâmetros livres e da variância residual.
- Covariância local calculada a partir do Jacobiano dos resíduos. A antiga
  pseudoinversa da Hessiana penalizada era singular pelo softmax e omitia o fator
  dois da Hessiana da soma de quadrados. Limites ativos, falta de posto ou
  convergência incompleta impedem apresentar erros-padrão como confiáveis.
- Mantida seleção pelo primeiro mínimo local. Candidatos com conjuntos distintos
  de observações retidas não são ordenados automaticamente, mesmo com o mesmo N.
  Falhas de convergência e critérios não finitos são excluídos da seleção.
- A primeira ordem monofásica pode ser reutilizada nas duas famílias sem novo
  ajuste, mas continua sendo considerada na comparação de ambas.
- A restrição de assíntota final zero foi retirada: ela não implementava uma
  cinética decrescente válida com essas equações e taxas positivas.
- Removidos scripts externos de badges. Mantidas referências por links.
- Exportação com dados brutos, candidatos, parâmetros numéricos, SE, resíduos,
  exclusões, opções, semente, horário UTC, SHA-256 da entrada e build. Strings
  não são convertidas em fórmulas ou links executáveis no Excel.
- Figuras são fechadas e seu uso é serializado para evitar interferência entre
  sessões. Limites por arquivo, por análise e de concorrência reduzem consumo
  acidental de recursos após a abertura do acesso.

As correções podem alterar erros-padrão, critérios e modelos selecionados.
Resultados históricos devem ser preservados com sua versão; não são reescritos.

## Instalação institucional

**O `git pull` deve ser feito na branch `main` do repositório público.** Ele
atualiza o código, mas não instala pacotes nem troca o Python do serviço.
Se o ambiente ainda utilizar Python 3.9, criar um novo ambiente Python 3.12,
instalar os requisitos e apontar o `ExecStart` do serviço para o novo executável.
Não basta reinstalar os requisitos dentro de um ambiente Python 3.9.

Exemplo de preparação, preservando o ambiente antigo para retorno:

```bash
git pull --ff-only origin main
python3.12 -m venv venv-py312
venv-py312/bin/python -m pip install --upgrade pip
venv-py312/bin/python -m pip install --upgrade -r requirements.txt
venv-py312/bin/python -m pip check
venv-py312/bin/python scripts/check_legacy_frontend.py
```

Ajustar o serviço para usar o caminho absoluto de `venv-py312/bin/python -m
streamlit run app.py`, mantendo diretório de trabalho, porta e proxy existentes.
Após editar a unidade: `systemctl daemon-reload` e
`systemctl restart streamlit-mono_polyauxic_kinetics`.
A informática deve executar essas etapas com as permissões administrativas
habituais; o programa não altera a unidade nem reinicia serviços automaticamente.

1. Confirmar Python 3.12, diretório do checkout e executável utilizado pelo
   serviço `streamlit-mono_polyauxic_kinetics`. O Python do shell pode ser diferente.
2. Preservar o checkout anterior e registrar `python -m pip freeze` do ambiente
   atualmente utilizado. Criar um ambiente separado se o Python antigo for <3.12.
3. Atualizar o checkout com a revisão revisada desta manutenção. Instalar com o
   **mesmo Python do serviço**, não com um `pip` global:

   ```bash
   /CAMINHO/DO/AMBIENTE/bin/python -m pip install --upgrade -r requirements.txt
   /CAMINHO/DO/AMBIENTE/bin/python -m pip check
   /CAMINHO/DO/AMBIENTE/bin/python scripts/check_legacy_frontend.py
   ```

4. Reiniciar o serviço após a instalação e conferir os logs e a página HTTPS.
   O acesso deve abrir diretamente. Validar o exemplo sintético, um upload
   autorizado e os downloads. Confirmar WebSocket e limites no proxy reverso.
5. Solicitar nova varredura externa do CSIRT no domínio. Se o navegador ou proxy
   mantiver assets antigos, conferir cache e a revisão realmente executada.
   **Uma atualização de requirements sem reiniciar não troca assets já servidos.**
6. Em caso de falha, restaurar checkout e ambiente anteriores como conjunto.
   Não apagar dados antigos nem reaproveitar segredos no repositório.

Este trabalho não comprova instalação no servidor ou liberação do chamado.
A notificação ao responsável e o encerramento dependem da verificação externa.

## Melhorias propostas

1. Adotar explicitamente um modelo para respostas decrescentes e tratar sinal,
   assíntotas e interpretação de taxas como parte do protocolo científico.
2. Validar o procedimento de outliers contra a referência metodológica desejada;
   o legado combina Charbonnier, MAD e BH e foi rotulado como aproximação.
3. Acrescentar perfil de verossimilhança/bootstrap para identificabilidade prática,
   intervalos de confiança e incerteza na seleção de fases.
4. Avaliar, como decisão metodológica separada, mínimo global versus primeiro
   mínimo local e a política de alternância AIC/AICc/BIC.
5. Se a demanda pública crescer, separar execução numérica em fila de processos
   com cancelamento e quotas por origem no proxy; atualmente o limite vale por
   processo e o orçamento é cooperativo dentro do otimizador.

## Fontes técnicas

- [Streamlit 1.67.0](https://github.com/streamlit/streamlit/releases/tag/1.67.0)
- [Lockfile JavaScript 1.67.0](https://github.com/streamlit/streamlit/blob/1.67.0/frontend/yarn.lock)
- [DOMPurify 3.4.17](https://github.com/cure53/DOMPurify/releases/tag/3.4.17)
- [Moment 2.31.0](https://github.com/moment/moment/releases/tag/2.31.0)
- [Equações de referência](https://doi.org/10.1007/s11538-026-01621-7)

## Verificação desta revisão

- **45 testes passaram** em Python 3.12: CSV/XLSX e limites, erros de entrada,
  recuperação de parâmetros em uma e duas fases, primeira ordem, mudança de
  unidades de tempo, remoção de outliers, critérios, concorrência, timeout,
  exportação e fluxo real de widgets com `streamlit.testing.v1.AppTest`.
- Upload → ajuste → gráficos → exportação conferidos em inglês, português e
  francês. Mudanças de apresentação não repetem o ajuste; novas condições e
  arquivos inválidos removem os resultados anteriores. Sessões são isoladas.
- Análise estática Ruff e verificação de whitespace concluídas sem erros.
- Verificador do JavaScript instalado: DOMPurify 3.4.17 e Moment 2.31.0 aprovados.
- Consulta OSV em 12/11/2026: nenhum alerta retornado para as 46 dependências do
  ambiente de validação e para Watchdog 7.1.0, específico de Linux/Windows.
  Isso representa a cobertura da base consultada, não prova ausência de falhas.
- Testes locais em macOS; a automação GitHub acrescenta execução nativa em Linux.
  A execução institucional e a varredura externa permanecem pendentes.

Os testes sintéticos exercitam os casos descritos. Não constituem validação
universal da identificabilidade, do número de fases ou do método de outliers.
