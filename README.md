Controllable Chain-of-Thought Compression in LLMs BUT from an energy point of view: Modelling the tradeoff accuracy vs power consumption of reasoning AI models in test time. 

The graphs you can see in this readme are the result of an average of three experiments, you can find the folders with the log files and everything the experiment is based on in the "final" folder

1) First I verified the consistency of the energy meaurment, remember that we have used 2 of them (out of 3 provides by grid5000, remember to say that PDU is not good and why):
   -WATTMETER -BMC (onboard sensor)
   Here we have in red the wattmeter measurment and in blu the BMC measurment, they are placed on different scales to be superimposed to show the Correlation.
   So starting from now all the energy measurment are plotted on the wattmeter's data since they are the best because they consider the whole machine and not
   only the energy consumed by the GPU.
   <img width="4200" height="2100" alt="graph_consistency_dual_axis_avg" src="https://github.com/user-attachments/assets/e055541b-7e33-4c52-a845-b683a3b704bf" />

2) Reproducibility of the experiment, here we have the figure 5 of the paper TokenSkip
   <img width="3000" height="2100" alt="graph_accuracy_tokens_connected_avg" src="https://github.com/user-attachments/assets/a49ac409-4ca7-4f3e-b201-3c6f7e68e04f" />
   Following there is the graph resulting from my data:
   <img width="3000" height="2100" alt="graph_accuracy_tokens_connected" src="https://github.com/user-attachments/assets/2ff7d020-2422-444a-b977-9979b8867c8d" />
   My experiments closely replicated the accuracy dynamics observed in the state of the art. As shown in the graph, model 14B maintains stable accuracy (green line) as
   the number of tokens decreases, while smaller models show progressive degradation, validating TokenSkip's effectiveness on large language models.
   <img width="5400" height="1800" alt="graph_token_reduction_trend_avg" src="https://github.com/user-attachments/assets/97f4d29e-0d83-4d88-a20a-b979a8bbaabc" />
   
3) Energy, remember that we used only the wattmeter measurment, the other one is only to check if the data were consistent.
   <img width="3000" height="2100" alt="graph_energy_savings_avg" src="https://github.com/user-attachments/assets/80a30b16-ef54-4a3b-a279-a6833d7daaef" />
   <img width="6000" height="1800" alt="graph_tradeoff_unified_avg" src="https://github.com/user-attachments/assets/fa0ab90b-243f-44ca-8ffc-e2dbe45771bf" />
   <img width="6000" height="1800" alt="graph_reasoning_vs_energy_avg" src="https://github.com/user-attachments/assets/4783597e-9eee-452f-93f6-95b548d07978" />
