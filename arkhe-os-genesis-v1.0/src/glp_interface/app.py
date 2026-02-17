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
from merkabah import MERKABAH7
from minoan import MinoanHardwareInterface, MinoanStateGrammar, MinoanApplications, MinoanNeuroethics

app = Flask(__name__)

# Configuração
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 128

# Modelos
glp_model = BCD_GLPLinearA(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
merkabah = MERKABAH7(linear_a_corpus=[], operator_profile={'intention': 'decode_linear_a'})
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
        'integrated_layers': ['GLP_BCD', 'MERKABAH-7', 'MINOAN_EXT'],
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
