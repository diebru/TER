TokenSkip: Controllable Chain-of-Thought Compression in LLMs BUT from an energy point of view: Modelling the tradeoff accuracy vs power consumption of reasoning AI models in test time. 
1) First I verified the consistency of the energy meaurment, remember that we have used 2 of them (out of 3 provides by grid5000, remember to say that PDU is not good and why):
   -WATTMETER -BMC (onboard sensor)
   Here we have in red the wattmeter measurment and in blu the BMC measurment, they are placed on different scales to be superimposed to show the Correlation.
   So starting from now all the energy measurment are plotted on the wattmeter's data since they are the best because they consider the whole machine and not
   only the energy consumed by the GPU.
   <img width="4200" height="2100" alt="graph_consistency_dual_axis" src="https://github.com/user-attachments/assets/c355960c-48c9-4097-9ef6-f350a542163f" />
2) Reproducibility of the experiment, here we have the figure 5 of the paper TokenSkip
   <img width="795" height="550" alt="image" src="https://github.com/user-attachments/assets/935949ac-f603-4c11-bd01-6af13d8c0a6a" />
   Following there is the graph resulting from my data:
   <img width="3000" height="2100" alt="graph_accuracy_tokens_connected" src="https://github.com/user-attachments/assets/2ff7d020-2422-444a-b977-9979b8867c8d" />
   My experiments closely replicated the accuracy dynamics observed in the state of the art. As shown in the graph, model 14B maintains stable accuracy (green line) as
   the number of tokens decreases, while smaller models show progressive degradation, validating TokenSkip's effectiveness on large language models.
   <img width="5400" height="1800" alt="graph_token_reduction_trend" src="https://github.com/user-attachments/assets/e0b0e172-1474-4d57-aa86-07b5381f58f2" />
3) Energy, remember that we used only the wattmeter measurment, the other one is only to check if the data were consistent.
   <img width="3000" height="2100" alt="graph_energy_savings" src="https://github.com/user-attachments/assets/c3bcf480-0a7e-464e-9f78-301f2275a607" />
   <img width="6000" height="1800" alt="graph_tradeoff_unified" src="https://github.com/user-attachments/assets/08624a8c-45cd-42d7-81a1-6a21a2b24b27" />
   <img width="6000" height="1800" alt="graph_reasoning_vs_energy" src="https://github.com/user-attachments/assets/cfa8f71d-91ed-492e-b51d-a796bbdc2ba6" />
