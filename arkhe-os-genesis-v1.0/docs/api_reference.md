# API Reference

## Arkhe Core (porta 8080)
- `GET /status` – retorna estado do nó (id, coherence, satoshi, handovers)
- `POST /handover` – realiza handover para outro nó (body: {to, payload})
- `GET /anticipate` – retorna predição de coerência futura

## GLP Server (porta 5000)
- `GET /status` – retorna camadas integradas e estado de coerência quântica.
- `POST /encode` – processa sequências de signos via arquitetura BCD e MERKABAH-7. (body: {sign_ids: [[...]]})
- `POST /steer` – aplica direção de conceito no manifold latente com auxílio do motor de metáfora.
- `POST /decode_tablet` – realiza decifração neuro-epigráfica com verificação ética. (body: {tablet_id, operator_caste})
- `POST /train_primordial` – executa treinamento GLP sem frameworks (NumPy/explicit gradients).
- `POST /observe_phi` – registra observações de alta coerência na camada Φ.
