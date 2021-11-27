#!/usr/bin/env python

# Imports

# Class ROM
class ROM:
    """
    Class containing the creation of the ROM. This, in turns, uses the database
    obtained at one operating condition and the physical model to obtain a reduced
    order model for the aerodynamics.
    In can be evolved in time to obtain a solution.
    It receives as input in the constructor the filename of the structural history
    file, the template name of the aerodynamic history file (i.e. if the history
    files are history_001.dat, history_002.dat, ecc.. the template name is history.dat),
    and the path to the folder where all the physical model files are contained.
    These are the deformation shape files and the normals file.
    """

    def __init__(self, filenameStru, filenameAero, pathToModel):
        self.data = database(filenameStru, filenameAero)
        self.deltaT = database.getDeltaT()
        self.model = physicalModel(pathToModel)
        __createABmatrices()


    def __createABmatrices(self):
        
        % Load the snapshot matrix.
% The matrix is evolving in time column-wise
SnapshotLoad = load(strcat(pwd,filesep,'V',num2str(Velocity),filesep,'Snapshot.mat'),'Snapshot');
Snapshot = SnapshotLoad.Snapshot;
clear SnapshotLoad

X = Snapshot(:,1:end-1);
Xp = Snapshot(:,2:end);

% center matrices
Xmean = mean(X,2);
X = X - Xmean;
Xp = Xp - Xmean;

% Same as before for the input snapshots
SnapshotLoad = load(strcat(pwd,filesep,'V',num2str(Velocity),filesep,'Snapshot_input.mat'));
Snapshot_input = SnapshotLoad.Snapshot_input;
clear SnapshotLoad
UPS = Snapshot_input(:,1:end-1);
UPSp = Snapshot_input(:,2:end);

% We can now stack the matrices together and perform the SVD. This will be
% truncated based on the automatic truncation explained in D. L. Donoho
% and M. Gavish, "The Optimal Hard Threshold for Singular Values is
% 4/sqrt(3)". The obatined SVD will be used to approximate the state-space
% matrices of the system. Note: they would have the same dimension as the
% state, in order to obtain a ROM, we project those matrices on the PODs
% obained with an SVD of the state only --> 2nd SVD in this script.
XModified = [X;UPS;UPSp];
[U,Sig,V] = svd(XModified,'econ');
figure
semilogy(Sig,'bo')
title('Singular values in descreasing order of importance of the SVD of the full Snapshot matrix')
thresh = optimal_SVHT_coef(size(Sig,2)/size(Sig,1),Sig);
rtil = length(find(diag(Sig)>thresh));
Util = U(:,1:rtil);
Sigtil = Sig(1:rtil,1:rtil);
Vtil = V(:,1:rtil);

[U,Sig,V] = svd(Xp,'econ');
figure
semilogy(Sig,'bo')
title('Singular values in descreasing order of importance of the SVD of the matrix Xprime')
thresh = optimal_SVHT_coef(size(Sig,2)/size(Sig,1),Sig);
r = length(find(diag(Sig)>thresh));
Uhat = U(:,1:r);
Sighat = Sig(1:r,1:r);
Vhat = V(:,1:r);

% Here we extract the matrices.
n = size(X,1);
q = size(UPS,1);
U_1 = Util(1:n,:);
U_2 = Util(n+1:n+q,:);
U_3 = Util(n+q+1:n+q+q,:);
Atil = Uhat'*(Xp)*Vtil*inv(Sigtil)*U_1'*Uhat;
Btil = Uhat'*(Xp)*Vtil*inv(Sigtil)*U_2';
Ftil = Uhat'*(Xp)*Vtil*inv(Sigtil)*U_3';

% Here we extract the eigenvalues for plotting purpose.
[DMtil,EIGtil] = eig(Atil);

figure
DiagonalEig=diag(EIGtil);
plot(real(DiagonalEig),imag(DiagonalEig),'gd')
title('Eigenvalues of the A matrix of the system')
hold on, grid on
theta = (0:1:100)*2*pi/100;
plot(cos(theta),sin(theta),'k--') % plot unit circle
for i=1:length(diag(EIGtil))
    if norm([real(DiagonalEig(i)),imag(DiagonalEig(i))])>1
        disp('Unstable system at Velocity:')
        disp(paramFSI.inputCreate.Velocity)
        plot(real(DiagonalEig(i)),imag(DiagonalEig(i)),'ro')
    end
    break
end
