% Simulation of Temporal Difference (TD) learning 
% adopted from a code by David S. Touretzky 
% available at: https://www.cs.cmu.edu/afs/cs/academic/class/15883-f15/matlab/td/

clc
clear
close all

%%

buffer_length   = 40 ;
num_stimuli     = 1 ;
Ntrials         = 200 ;
pEnv            = 1 ;

buffer  = zeros(buffer_length,num_stimuli);
reward  = zeros(buffer_length,1);
history = zeros(buffer_length,4);  % [V, hV, R, delta]

W       = buffer;
V       = 0 ;
Gamma   = 1 ;
eta     = 0.2 ;

% set up stimulus and reward patterns
stimpat         = buffer;
stimpat(buffer_length/8,1)    = 1 ;

epoch   = 0;
pat     = Inf;

for cntN = 1:Ntrials
    reward(buffer_length*5/8)     = pEnv > rand ;
    for cnt_buff=1:buffer_length
        i = (cntN-1)*buffer_length + cnt_buff ;
        
        % Main loop for TD learning
        pat     = pat+1;
        if pat > buffer_length
          epoch = epoch + 1;
          pat   = 1;
          buffer(:)     = 0;
          history(:)    = 0;
          V     = 0 ;
        end

        % shift new stimulus into buffer
        hbuff               = buffer;
        buffer(2:end,:)     = buffer(1:(end-1),:);
        buffer(1,:)         = stimpat(pat,:);
        R                   = reward(pat);

        % TD learning rule
        hV                  = V;
        V                   = dot(W(:),buffer(:));
        delta               = R + Gamma*V - hV;
        history(2:end,:)    = history(1:(end-1),:);
        history(1,:)        = [V hV R delta];
        W(:)                = W(:) + eta*delta*hbuff(:);
        hhistory(:,:,i)     = history ; 
    end
    VRL(cntN) = mean(history(history(:,1)~=0,1)) ;
end

%%

figure(1)
for cntN = 1:10
    subplot(2,5,cntN)
    hold on
    plot(hhistory(end:-1:1,1,cntN*buffer_length*Ntrials/10), 'r-', 'linewidth', 3)
    plot(hhistory(end:-1:1,4,cntN*buffer_length*Ntrials/10), 'b-', 'linewidth', 3)
    stem(reward, 'k','LineStyle','--', 'marker','none')
    stem(stimpat, 'k','LineStyle','--', 'marker','none')
    axis([0 buffer_length 0 1])
    if cntN==1
        legend({'V', '\delta'})
    end
    set(gca,'FontName','Helvetica','FontSize',23,'FontWeight','normal','LineWidth',2,'yTick',-0.:0.25:1,'Xtick',0:buffer_length/4:buffer_length)
    set(gca, 'tickdir', 'out')
end

FigW = 4*5;
FigH = 3*2;
set(gcf,'units','centimeters')
set(gcf,'position',[10,10,3*FigW,3*FigH],'PaperSize',[FigW FigH],'PaperPosition',[0,0,FigW,FigH],'units','centimeters');  

% close all; plot(VRL)

