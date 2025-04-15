from pathlib import Path

import torch
from transformers import AutoProcessor, HubertModel

class HubertEncoder(torch.nn.Module):
    def __init__(self, model_name = 'facebook/hubert-large-ls960-ft'):
        super().__init__()

        if not Path(model_name).exists():
            self.download_from_hub(model_name)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)

        self.sampling_rate = 16000
        self.output_dim = 1024
        self.hop_size_in_ms = 20 # (ms) length: 320 
        self.max_length = int(10*self.sampling_rate)


    def download_from_hub(self, model_name: str, output_root: str = "."):
        # Saving to local to avoid multiprocessing issues
        output_dir = Path(output_root) / model_name
        self.processor =  AutoProcessor.from_pretrained(model_name)#"facebook/hubert-large-ls960-ft")#model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.processor.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

    # def forward(self, audio: torch.Tensor):
    #     assert isinstance(audio, torch.Tensor)  
    #     if audio.ndim == 1:
    #         audio = audio.unsqueeze(0)
        
    #     device = audio.device
    #     audio = audio.cpu().numpy()

    #     # Feature extraction
    #     features = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")
    #     features["input_values"] = features["input_values"].to(device)
    #     return self.model(features['input_values']).last_hidden_state

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)  
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        device = audio.device

        if audio.shape[-1] > self.max_length:
            output = []
            for chunk in audio.split(self.max_length, dim=-1):
                if chunk.shape[-1] < self.sampling_rate:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, self.sampling_rate - chunk.shape[-1]))
                chunk = chunk.cpu().numpy()
                tmp_features = self.processor(chunk, sampling_rate=self.sampling_rate, return_tensors="pt")
                tmp_features["input_values"] = tmp_features["input_values"].to(device)
                tmp_output = self.model(tmp_features['input_values']).last_hidden_state
                output.append(tmp_output)
            output = torch.cat(output, dim = 1)
        else:
            # Feature extraction
            audio = audio.cpu().numpy()
            features = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")
            features["input_values"] = features["input_values"].to(device)
            output = self.model(features['input_values']).last_hidden_state
        return output



if __name__ == '__main__':
    from xares.audio_encoder_checker import check_audio_encoder
    encoder = HubertEncoder()
    out = encoder(torch.rand(3, 16000));
    print(out.shape)
    assert check_audio_encoder(encoder)