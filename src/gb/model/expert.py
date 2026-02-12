import torch


class Expert(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, ep_group: torch.ProcessGroupNCCL):
        self.model = model
        self.process_group = ep_group
        
        self.loss_weights = self.make_loss_weights()
        
    def make_loss_weights() -> torch.Tensor:
        self.weights = torch.zeros((84,), dtype=self.model.dtype, device=self.model.device ) # this is hard-coded to the 84 variables we have agreed on. Not pretty, but better than string comparisons 

        match self.pg.rank():
            case 0:
                # surface variable expert: 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'sea_surface_temperature','total_precipitation'
                self.weights[0:6] = 1.0
            case 1:
                # wind expert: u, v and w at all pressure levels
                self.weights[45:] = 1.0
            case 2:
                # thermodynamic expert: geopotential, temperature and specific humidity at all pressure levels
                self.weights[6:45] = 1.0
            case _:
                # universal expert, all variables; default
                self.weights = self.ones_like(self.weights)
            
        return self.weights
    
    def get_loss_weights() -> torch.Tensor:
        return self.loss_weights
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
