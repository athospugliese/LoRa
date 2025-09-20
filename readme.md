# TinyLlama Fine-tuning com LoRA

Este repositório contém o código para fine-tuning do modelo TinyLlama-1.1B utilizando a técnica LoRA (Low-Rank Adaptation) para criar um assistente de chat com recursos computacionais limitados.

## 🚀 Visão Geral

O projeto demonstra como adaptar o modelo TinyLlama-1.1B usando Parameter-Efficient Fine-Tuning (PEFT) com o dataset Guanaco. Este método permite fine-tuning de modelos de linguagem grandes mesmo com hardware modesto.
## 🔗 Modelo no Hugging Face Hub

O modelo treinado está disponível para uso:

**[athospugliese/llama-1.1B-chat-guanaco](https://huggingface.co/athospugliese/llama-1.1B-chat-guanaco)**

Você pode usar o modelo diretamente através da API do Hugging Face ou baixá-lo para uso local.

## 📋 Requisitos

Para executar este notebook, você precisará:

- Python 3.x
- PyTorch
- Transformers
- PEFT
- TRL (Transformer Reinforcement Learning)
- Datasets
- BitsAndBytes (para quantização opcional)

Instale as dependências com:
```bash
pip install transformers datasets evaluate peft trl bitsandbytes
```

## 🔧 Configuração do Ambiente
O notebook foi testado com uma GPU NVIDIA T4 (16GB de VRAM). O código inclui otimizações para funcionar em ambientes com restrições de memória.
## 📊 Dataset

Usamos o dataset Guanaco-Llama2-1k, uma coleção de 1.000 exemplos de instruções de alta qualidade formatados para modelos Llama 2. O dataset está disponível em:

**[mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)**

## 💻 Código

O notebook contém código para:

1. **Configuração inicial:** Importação de bibliotecas e definição de variáveis
2. **Carregamento do modelo e tokenizer:** Configuração do TinyLlama base
3. **Teste inicial do modelo:** Verificação de inferência antes do fine-tuning
4. **Configuração LoRA:** Definição de hiperparâmetros PEFT/LoRA
5. **Configuração de treinamento:** Parâmetros de treinamento otimizados
6. **Treinamento:** Fine-tuning do modelo
7. **Salvamento do modelo:** Armazenamento do adaptador LoRA e tokenizer
8. **Avaliação:** Teste do modelo treinado

### Hiperparâmetros LoRA

```python
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

### Hiperparâmetros de Treinamento

```python
training_params = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine"
)
```

## 📝 Formato de Prompt

O modelo espera prompts no formato:

```
<s>[INST] sua pergunta ou instrução aqui [/INST]
```

## 🔍 Exemplos

### Exemplo 1: Conhecimento Histórico

**Prompt:** Who is Napoleon Bonaparte?

**Resposta:**
> Napoleon was the French version of his name. He was a famous general and statesman who led France to victory in several wars, including the Napoleonic Wars (1792–1815). In 1804 he declared himself emperor of the French Empire.

### Exemplo 2: Explicação Científica

**Prompt:** Explain quantum computing in simple terms.

**Resposta:**
> Quantum computers are machines that use quantum mechanics to perform calculations faster than conventional computers and allow for the creation of complex algorithms, such as those used by AI systems like Google's DeepMind.
>
> Quantum computing uses qubits, or "qubit states," which can be in a superposition state (i.e., they can have both an up and down spin) rather than being either on or off. This allows for more efficient processing, since it enables the computer to take advantage of the properties of quantum physics instead of relying solely on classical logic.

## 💾 Estrutura do Modelo Treinado
Após o treinamento, o modelo é salvo no diretório especificado (`llama-1.1B-chat-guanaco`) com a seguinte estrutura:

```
llama-1.1B-chat-guanaco/
├── README.md
├── adapter_config.json
├── adapter_model.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
```

Os checkpoints intermediários são salvos no diretório `results/`.

## 🚀 Como Usar o Modelo Treinado

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Carregar a configuração PEFT
config = PeftConfig.from_pretrained("athospugliese/llama-1.1B-chat-guanaco")

# Carregar o modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Aplicar o adaptador LoRA
model = PeftModel.from_pretrained(base_model, "athospugliese/llama-1.1B-chat-guanaco")

# Carregar o tokenizer
tokenizer = AutoTokenizer.from_pretrained("athospugliese/llama-1.1B-chat-guanaco")
tokenizer.pad_token = tokenizer.eos_token

# Função para gerar respostas
def generate_response(prompt, max_length=200):
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()
    return response

# Exemplo
prompt = "What is the meaning of life?"
print(generate_response(prompt))
```

## ⚠️ Limitações

- Modelo de tamanho pequeno (1.1B parâmetros), com capacidades de raciocínio limitadas
- Pode produzir informações factuais incorretas
- Conhecimento limitado aos dados de treinamento
- Pode apresentar vieses presentes nos dados
- Não otimizado para tarefas especializadas

## 👨‍💻 Autor

**Athos Pugliese**

- GitHub: [github.com/athospugliese](https://github.com/athospugliese)
- Email: athospugliesedev@gmail.com
- Hugging Face: [athospugliese](https://huggingface.co/athospugliese)

## 📜 Licença
MIT License

## 🙏 Agradecimentos

- **TinyLlama Project** pelo modelo base
- **PEFT Library** por tornar possível o fine-tuning eficiente
- **TRL Library** pelo framework de treinamento
- **Hugging Face** pelo ecossistema de ferramentas


---

Este projeto foi desenvolvido com o objetivo educacional de demonstrar técnicas de fine-tuning de modelos de linguagem com recursos limitados.