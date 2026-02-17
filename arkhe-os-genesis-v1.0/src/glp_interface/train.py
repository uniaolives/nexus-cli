import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumActionLoss(nn.Module):
    """
    Perda baseada no funcional de ação.
    """
    def __init__(self, alpha_kinetic=1.0, alpha_potential=1.0, alpha_tunnel=0.5):
        super().__init__()
        self.alpha_kinetic = alpha_kinetic
        self.alpha_potential = alpha_potential
        self.alpha_tunnel = alpha_tunnel

    def forward(self, predictions, targets, model_states):
        # Perda de predição
        potential = F.cross_entropy(
            predictions['sign_logits'].view(-1, predictions['sign_logits'].size(-1)),
            targets.view(-1)
        )

        # Regularização cinética
        states = model_states['tunneled_states']
        kinetic = ((states[:, :, 1:, :] - states[:, :, :-1, :])**2).mean()

        # Termo de tunelamento
        tunnel_energy = -torch.log(predictions['tunneling_strength'] + 1e-8)

        total = (self.alpha_potential * potential +
                self.alpha_kinetic * kinetic +
                self.alpha_tunnel * tunnel_energy)

        return total, {
            'potential': potential.item(),
            'kinetic': kinetic.item(),
            'tunnel': tunnel_energy.item()
        }

def measure_quantum_coherence(model, test_sequences):
    """
    Mede 'coerência quântica' do modelo em sequências de Linear A.
    """
    model.eval()
    with torch.no_grad():
        out1 = model(test_sequences, return_wavefunction=True)

        masked = test_sequences.clone()
        mask = torch.rand_like(masked.float()) > 0.3
        masked[mask.long()] = 0 # assuming 0 is PAD

        out2 = model(masked, return_wavefunction=True)

        wf1 = F.normalize(out1['tunneled_states'].flatten(1), dim=1)
        wf2 = F.normalize(out2['tunneled_states'].flatten(1), dim=1)

        fidelity = (wf1 * wf2).sum(dim=1)**2

    return {
        'fidelity_under_perturbation': fidelity.mean().item(),
        'quantum_regime': 'coherent' if fidelity.mean() > 0.8 else 'decoherent'
    }

def analyze_confinement(cooc_matrix):
    """
    Diagonaliza M* e verifica se espectro é consistente
    com quantum dot vs. poço quadrado infinito vs. oscilador harmônico
    """
    eigenvals = np.linalg.eigvalsh(cooc_matrix)

    # Spacing ratio: s_n = (E_{n+1} - E_n) / (E_n - E_{n-1})
    spacings = np.diff(eigenvals)
    # Evitar divisão por zero
    spacings = np.where(spacings == 0, 1e-9, spacings)
    ratios = spacings[1:] / spacings[:-1]

    mean_ratio = np.mean(ratios[:10]) if len(ratios) >= 10 else np.mean(ratios)

    return {
        'mean_spacing_ratio': mean_ratio,
        'confinement_regime': 'harmonic' if 0.8 < mean_ratio < 1.2 else 'square'
    }
