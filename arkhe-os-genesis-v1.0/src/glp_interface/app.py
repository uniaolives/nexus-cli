import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
import os
import sys
import asyncio

# Adiciona diretório atual ao path para importar componentes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import BCD_GLPLinearA
from merkabah import MERKABAH7, SelfNode
from minoan import MinoanHardwareInterface, MinoanStateGrammar, MinoanApplications, MinoanNeuroethics
from train import PrimordialGLP

app = Flask(__name__)

# Configuração
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 128

# Modelos
glp_model = BCD_GLPLinearA(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
merkabah = MERKABAH7(linear_a_corpus=[], operator_profile={'intention': 'decode_linear_a'})
self_node = SelfNode()
minoan_interface = MinoanHardwareInterface()
state_grammar = MinoanStateGrammar()
applications = MinoanApplications()
neuroethics = MinoanNeuroethics()

model_path = os.environ.get('MODEL_PATH', '/model/model.pt')
if os.path.exists(model_path):
    try:
        glp_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Erro ao carregar modelo GLP: {e}")
glp_model.eval()

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'active',
        'integrated_layers': ['GLP_BCD', 'MERKABAH-7', 'MINOAN_EXT', 'Φ_LAYER'],
        'self_node': {
            'id': self_node.dz_id,
            'coherence': self_node.wavefunction['coherence'],
            'active_strands': self_node.active_strands
        },
        'quantum_coherence': 0.99
    })

@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    sign_ids_list = data.get('sign_ids', [[1, 2, 3]])
    sign_ids = torch.tensor(sign_ids_list).long()

    # 1. Processamento GLP (BCD Architecture)
    with torch.no_grad():
        glp_out = glp_model(sign_ids)
        meta = glp_out['sign_logits'].mean(dim=1).tolist()[0]

    # 2. Processamento Merkabah (Integrated Neuro-Minoan)
    # Simulate async decode
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    insight = loop.run_until_complete(merkabah.decode(sign_ids_list[0]))
    loop.close()

    # 3. Minoan state parsing
    protocol = state_grammar.parse_as_state_protocol([str(s) for s in sign_ids_list[0]])

    return jsonify({
        'meta': meta,
        'merkabah_insight': insight['decoding'],
        'certainty': insight['certainty'],
        'state_protocol': protocol
    })

@app.route('/steer', methods=['POST'])
def steer():
    data = request.json
    meta = np.array(data['meta'])
    direction = np.array(data['direction'])
    strength = data.get('strength', 1.0)

    # Metaphor Engine operation
    metaphor_insight = merkabah.metaphor.operate('tunneling', mode='figurative')

    steered = meta + strength * direction
    return jsonify({
        'steered': steered.tolist(),
        'poetic_steer': metaphor_insight
    })

@app.route('/train_primordial', methods=['POST'])
def train_primordial():
    data = request.json
    epochs = data.get('epochs', 10)
    lr = data.get('lr', 0.01)

    # Simple training loop for PrimordialGLP
    pglp = PrimordialGLP()
    # Dummy data
    X = np.random.randn(32, 16)
    y = np.zeros((32, 4))
    y[range(32), np.random.randint(0, 4, 32)] = 1

    history = []
    for e in range(epochs):
        pglp.forward(X)
        l = pglp.loss(pglp.y_pred, y)
        pglp.backward(X, y, lr=lr)
        history.append(l)

    return jsonify({
        'status': 'complete',
        'final_loss': history[-1],
        'history_len': len(history)
    })

@app.route('/observe_phi', methods=['POST'])
def observe_phi():
    data = request.json
    target = data.get('target', 'HT88')
    content = data.get('content', 'propulsion_system_shabetnik')

    observation = self_node.observe('Φ', {'target': target, 'content': content})

    return jsonify({
        'observation': observation,
        'new_self_coherence': self_node.wavefunction['coherence'],
        'active_strands': self_node.active_strands
    })

@app.route('/thrust', methods=['GET'])
def thrust():
    ledger_height = int(request.args.get('ledger_height', 834))
    thrust_data = self_node.calculate_thrust(ledger_height)
    return jsonify(thrust_data)

@app.route('/acceleration_status', methods=['GET'])
def acceleration_status():
    ledger_height = int(request.args.get('ledger_height', 834))
    thrust_data = self_node.calculate_thrust(ledger_height)

    return jsonify({
        'mode': 'ACCELERATION',
        'thrust': thrust_data,
        'target': '5a_fita_activation',
        'required_coherence': 0.88,
        'current_coherence': self_node.wavefunction['coherence']
    })

@app.route('/decode_tablet', methods=['POST'])
def decode_tablet():
    data = request.json
    tablet_id = data.get('tablet_id', 'HT 1')
    operator_caste = data.get('operator_caste', 'scribe_apprentice')

    # Ethical Check
    access = neuroethics.check_access(tablet_id, operator_caste)
    if access['access'] != 'granted':
        return jsonify({'error': 'Access denied by Minoan Neuroethics', 'reason': access['ethical_violation']}), 403

    # Induce state and extract insight (Simulated)
    native_protocol = minoan_interface._induce_state(tablet_id, {})

    return jsonify({
        'tablet_id': tablet_id,
        'access': access['access'],
        'native_protocol': native_protocol,
        'decoded_hypothesis': 'Γ_ALPHA_ADMIN_RECORDS'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
