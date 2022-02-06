function plot_state(particles, timestep,X,Xr,Yr,BestPart)
    % Visualizes the state of the particles
    
     clf;
    hold on
    grid on
hold on
    plot(Xr,Yr,'ob')
    % Plot the particles
    ppos = [particles.pose];
    plot(ppos(1,:), ppos(2,:), 'g.', 'markersize', 10, 'linewidth', 3.5);
% hold on
%     plot(ppos(1,:), ppos(2,:), 'g.', 'markersize', 10, 'linewidth', 3.5);
hold on
    plot(X(1,1:timestep),X(2,1:timestep),'.k')
%     hold on
%     plot(Xr,Yr,'ob')
try
     hold on
    plot(BestPart(1,:),BestPart(2,:),'.r')
end
    xlim([-6, 16])
    ylim([-6, 16])
    hold off

    % dump to a file or show the window
    %window = true;
    %window = false;
    if 1 %window
      hold on
      drawnow;
      pause(0.5);
    else
%       figure(1, "visible", "off");
%       filename = sprintf('../plots/pf_%03d.png', timestep);
%       print(filename, '-dpng');
    end

end
