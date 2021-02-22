import numpy as np
import matplotlib.pyplot as plt
import Data_Unfitted as Data

################################################################################
alignments = 10
chi2_cut = np.logspace(4,2.7,alignments)
markers=['o','^','x','s','p','h','D']
col=['#fbcf36','#ed4c1c','#9c7e70','#5ac2f1','#11776c','#e0363a','#6a1c10']
################################################################################

plt.style.use('bmh')
hit_data = Data.hit_data

cnt=0

# Initialize list for plotting with old offsets {{{
posx, posy, dposx, dposy = [], [], [], []
for i in range(7):
    posx.append([])
    posy.append([])
    dposx.append([])
    dposy.append([])
    dposx[i].append(0)
    dposy[i].append(0)

# Old offset 
posx[0].append(0)
posx[1].append(-10.89)
posx[2].append(-6.06)
posx[3].append(0.41)
posx[4].append(25.89)
posx[5].append(28.06)
posx[6].append(79.91)
posy[0].append(0)
posy[1].append(46.93)
posy[2].append(-11.77)
posy[3].append(-6.92)
posy[4].append(-8.00)
posy[5].append(34.12)
posy[6].append(19.14)

#posx[0].append(0)
#posx[1].append(-10.89)
#posx[2].append(-6.06)
#posx[3].append(0.00684)
#posx[4].append(27.83858)
#posx[5].append(28.06)
#posx[6].append(79.91)
#posy[0].append(0)
#posy[1].append(46.93)
#posy[2].append(-11.77)
#posy[3].append(-4.79166)
#posy[4].append(-8.38913)
#posy[5].append(34.12)
#posy[6].append(19.14)

# }}}

newchi2cut = 0 # adjust chi2 cut after first alignment
for a in range(alignments):

    # Count how many tracks are used for alignment after chi2cut
    cnt_track=0
    # Show chi2 distro
    chi2_array = []

    # Fit linear track
    N = len(hit_data) # Number of Tracks

    print('Tracking...') # {{{
    for track in range(N):
        
        # Only use plane 4 & 5 for tracking 
        planes_used = [1,4]
        plane1, plane2 = planes_used[0], planes_used[1]
        # Remember, which planes are involved, for residuals
        planes_involved = []
        std_hit = []

        # Apply initial alignment for these planes
        for plane in planes_used:
            hit_data[track][plane]["XC"]-=posx[plane][a]
            hit_data[track][plane]["YC"]-=posy[plane][a]

        # Apply alignment for the rest of the planes
        for plane in range(7):
            if (plane in planes_used) or (hit_data[track][plane]["XC"] == -1):
                continue
            planes_involved.append(plane)
            std_hit.append(hit_data[track][plane]['sig'])
            hit_data[track][plane]["XC"]-=posx[plane][a]
            hit_data[track][plane]["YC"]-=posy[plane][a]


        # Use position of hits on planes to create the track
        x1 = np.array([hit_data[track][plane1]["XC"],
                hit_data[track][plane1]["YC"],
                plane1*1024*2/3])
        x2 = np.array([hit_data[track][plane2]["XC"],
                hit_data[track][plane2]["YC"],
                plane2*1024*2/3])

        # Calculate the residuals for all other planes
        d = []
        for plane in planes_involved:
            x0 = np.array([
                hit_data[track][plane]["XC"],
                hit_data[track][plane]["YC"],
                plane*1024*2/3])
            # Solve for the point in plane of Track z = a+mb
            lbda = (x0[2] - x2[2])/(x2-x1)[2] # m = (z-a)/(b)
            xz = x2+lbda*(x2-x1) # find point in axis that lies in the same plane
            hit_data[track][plane]["resx"] = (x0-xz)[0]
            hit_data[track][plane]["resy"] = (x0-xz)[1]
            d.append(np.linalg.norm(xz-x0))

        # From there, calculate chi2 to determine the goodness of the fit
        chi2 = 0
        for entry in range(len(d)):
            chi2 += d[entry]**2/std_hit[entry]**2

        #plot distro
        if chi2 <= chi2_cut[cnt]:
            chi2_array.append(chi2)
        hit_data[track]["chi2"] = chi2
    # }}}

    #plt.hist(chi2_array,40)
    #plt.show()

    # Create Dictionary for Residuals
    Res = {}
    for plane in range(7):
        Res[plane] = {}
        Res[plane]['x'] = []
        Res[plane]['y'] = []

    # Fill Dictionary
    print('Calculating Residuals...')
    for i in range(N):
        if (hit_data[i]['chi2'] >= chi2_cut[cnt]): continue
        cnt_track+=1
        for plane in range(7):
            if (plane in planes_used) or (hit_data[i][plane]['XC'] == -1): continue
            Res[plane]['x'].append(hit_data[i][plane]['resx'])
            Res[plane]['y'].append(hit_data[i][plane]['resy'])

    # Calculate mean of residual (roughly)
    OffsetX, OffsetY, dOffsetX, dOffsetY = [], [], [], []
    for plane in range(7):
        if plane in planes_used:
            OffsetX.append(0)
            OffsetY.append(0)
            dOffsetX.append(0)
            dOffsetY.append(0)
            continue
        OffsetX.append(np.mean(Res[plane]['x']))
        OffsetY.append(np.mean(Res[plane]['y']))
        dOffsetX.append(np.std(Res[plane]['x']))
        dOffsetY.append(np.std(Res[plane]['y']))

    # Append changes to plot array
    print('{} Tracks survived the chi2 cut of {}'.format(cnt_track,chi2_cut[cnt]))
    for plane in range(7):
        print("Offset plane {}: x = {} y = {}".format(
            plane,np.round(OffsetX[plane],4),np.round(OffsetY[plane],4)))
        posx[plane].append(OffsetX[plane])
        posy[plane].append(OffsetY[plane])
        dposx[plane].append(dOffsetX[plane])
        dposy[plane].append(dOffsetY[plane])
    cnt+=1

# So far, only the new offsets have been written to posx... need to translate
for a in range(alignments):
    for plane in range(7):
        posx[plane][a+1]+=posx[plane][a]
        posy[plane][a+1]+=posy[plane][a]
for a in range(alignments+1):
    for plane in range(7):
        posx[plane][a]=-29.24*posx[plane][a]
        posy[plane][a]=-26.88*posy[plane][a]

for plane in range(7):
    print('Plane {} X: {}'.format(plane,np.round(posx[plane],2)))
    print('Plane {} Y: {}'.format(plane,np.round(posy[plane],2)))

# Plot section
print('Plotting...')
xaxis = np.arange(alignments+1)
plt.figure(figsize=(5,5))
plt.xlabel('Alignment iterations')
plt.ylabel('Plane position in X [um]')
for plane in range(7):
    if plane in planes_used: continue
    plt.errorbar(xaxis,posx[plane],yerr=dposx[plane], label='Plane {}'.format(plane),
            linewidth=1,marker=markers[plane],color='black',capsize=3,mfc=col[plane])
plt.legend(loc=2,bbox_to_anchor=(.9, .5))
plt.tight_layout()

plt.figure(figsize=(4.2,5))
plt.xlabel('Alignment iterations')
plt.ylabel('Plane position in Y [um]')
for plane in range(7):
    if plane in planes_used: continue
    plt.errorbar(xaxis,posy[plane],yerr=dposy[plane], label='Plane {}'.format(plane),
            linewidth=1,marker=markers[plane],color='black',capsize=3,mfc=col[plane])
#plt.legend(loc=2,bbox_to_anchor=(.9, .5))
plt.tight_layout()
plt.show()

fx = open("Data_Fitted_X.py","w")
fx.write("hit_data = "+str(hit_data))
fx.close
