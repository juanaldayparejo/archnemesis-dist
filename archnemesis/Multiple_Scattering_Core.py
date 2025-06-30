import os
import numpy as np
from numba import njit,prange


# @njit(fastmath=True, error_model='numpy')
# def phase1(calpha, iscat, cons, icons=0, icont=None, ncont=None, vwave=None, pfunc = None, xmu = None):
#     pi = np.pi
#     calpha = min(max(calpha, -1.0), 1.0)
#     if iscat == 0:
#         p = 0.75 * (1.0 + calpha * calpha)
#     elif iscat == 1:
#         p = 1.0
#     elif iscat == 2:
#         f1 = cons[0]
#         f2 = 1.0 - f1
#         hg11 = 1.0 - cons[1] * cons[1]
#         hg12 = 2.0 - hg11
#         hg21 = 1.0 - cons[2] * cons[2]
#         hg22 = 2.0 - hg21
#         p = f1 * hg11 / np.sqrt(hg12 - 2.0 * cons[1] * calpha) ** 3 + f2 * hg21 / np.sqrt(hg22 - 2.0 * cons[2] * calpha) ** 3
#     elif iscat == 3:
#         cons[0] = 0.25 / pi
#         p = 0.0
#         xf = calpha * calpha
#         for k in range(1, (icons + 1) // 2 + 1):
#             n0 = 2 * k - 1
#             n1 = 2 * k
#             x0 = 1.0
#             x1 = calpha
#             pa0 = 0.0
#             pa1 = 0.0
#             for i in range(1, k + 1):
#                 pa0 += x0 * jcoef[i - 1, n0 - 1] / jdiv[k - 1]
#                 pa1 += x1 * jcoef[i - 1, n1 - 1] / jdiv[k - 1]
#                 x0 *= xf
#                 x1 *= xf
#             p += cons[n0] * pa0 + cons[n1] * pa1
#     elif iscat == 4:
#         p = np.interp(calpha,xmu,pfunc)
        
#     else:
#         print('Error invalid scattering option.')
#         return None
#     p /= 4.0 * pi
#     return p

@njit
def phasint2_ref(nphi, ic, nmu, mu, iscat, cons, ncons, icont, ncont, pfunc, xmu):
    '''
    Function to compute the phase matrix for a Hapke surface
    '''
    
    pi = np.pi
    dphi = 2.0 * pi / nphi
    ntheta = nmu*nmu*(nphi+1)
    
    #Defining scattering angles
    ix = 0
    cpl = np.zeros(ntheta)
    cmi = np.zeros(ntheta)
    for j in range(nmu):
        for i in range(nmu):
            sthi = np.sqrt(1.0-mu[i]*mu[i]) #sin(theta[i])
            sthj = np.sqrt(1.0-mu[j]*mu[j]) #sin(theta[i])
            for k in range(nphi+1):
                phi = k*dphi
                
                #Calculating cos(alpha)
                cpl[ix] = sthi*sthj*np.cos(phi) + mu[i]*mu[j]
                cmi[ix] = sthi*sthj*np.cos(phi) - mu[i]*mu[j]
                
                if(cpl[ix]>1.):
                    cpl[ix] = 1.
                if(cpl[ix]<-1.):
                    cpl[ix] = -1.
                if(cmi[ix]>1.):
                    cmi[ix] = 1.
                if(cmi[ix]<-1.):
                    cmi[ix] = -1.
                    
                ix += 1
        
    #Calculating the phase function at the scattering angles
    if iscat == 0: #Rayleigh scattering
        ppl = 0.75 * (1.0 + cpl * cpl)/ (4 * np.pi)
        pmi = 0.75 * (1.0 + cmi * cmi)/ (4 * np.pi)
    else:
        ppl = np.interp(cpl, xmu, pfunc) #phase1(cpl, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)
        pmi = np.interp(cmi, xmu, pfunc) #phase1(cmi, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)        
        
    #Integrating the phase function over the azimuth direction
    pplpl = np.zeros((nmu,nmu))
    pplmi = np.zeros((nmu,nmu))
    ix = 0
    for j in range(nmu):
        for i in range(nmu):
            for k in range(nphi+1):
                phi = k*dphi
                    
                plx = ppl[ix] * np.cos(ic*phi)
                pmx = pmi[ix] * np.cos(ic*phi)
                
                wphi = 1.*dphi
                if k == 0:
                    wphi = 0.5*dphi
                elif k == nphi:
                    wphi = 0.5*dphi
                
                if ic == 0:
                    wphi = wphi / (2.*np.pi)
                else:
                    wphi = wphi / np.pi
                
                pplpl[i,j] += wphi * plx
                pplmi[i,j] += wphi * pmx
                
                ix += 1
                
    return (pplpl, pplmi)
                

@njit
def phasint2(nphi, ic, nmu, mu, iscat, cons, ncons, icont, ncont, pfunc, xmu):
    
    pi = np.pi
    dphi = 2.0 * pi / nphi

    phi = np.arange(nphi + 1) * dphi
    sth = np.sqrt(1.0 - mu**2)
    mu2d = mu[:, None]
    
    cos_ic_phi = np.cos(ic * phi)
    cos_phi_grid = np.cos(phi)

    # Broadcasting
    sth_sth = sth[:, None] * sth[None, :]
    mu_mu = mu2d * mu2d.T
    
    cpl = sth_sth[:, :, None] * cos_phi_grid + mu_mu[:, :, None]
    cmi = sth_sth[:, :, None] * cos_phi_grid - mu_mu[:, :, None]
    #Cutting out phase1 for now

    if iscat == 0:
        pl = 0.75 * (1.0 + cpl * cpl)/ (4 * np.pi)
        pm = 0.75 * (1.0 + cmi * cmi)/ (4 * np.pi)
    elif iscat == 2:
        f1 = pfunc[0]
        f2 = 1.0 - f1
        hg11 = 1.0 - pfunc[1] * pfunc[1]
        hg12 = 2.0 - hg11
        hg21 = 1.0 - pfunc[2] * pfunc[2]
        hg22 = 2.0 - hg21
        pl = f1 * hg11 / np.sqrt(hg12 - 2.0 * pfunc[1] * cpl) ** 3 + f2 * hg21 / np.sqrt(hg22 - 2.0 * pfunc[2] * cpl) ** 3
        pm = f1 * hg11 / np.sqrt(hg12 - 2.0 * pfunc[1] * cmi) ** 3 + f2 * hg21 / np.sqrt(hg22 - 2.0 * pfunc[2] * cmi) ** 3
        pl /= 4*np.pi
        pm /= 4*np.pi
    else:
        pl = np.interp(cpl, xmu, pfunc) #phase1(cpl, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)
        pm = np.interp(cmi, xmu, pfunc) #phase1(cmi, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)        
        
    wphi = np.full(phi.shape, dphi)
    wphi[0]  = 0.5 * dphi
    wphi[-1] = 0.5 * dphi
    
    if ic == 0:
        wphi /= (2.0 * pi)
    else:
        wphi /= pi
    
    wphi = wphi[None, None, :]
    
    plx = pl * cos_ic_phi
    pmx = pm * cos_ic_phi
    
    pplpl = np.sum(wphi * plx, axis=2)
    pplmi = np.sum(wphi * pmx, axis=2)

    return (pplpl, pplmi)


@njit(fastmath=True, error_model='numpy')
def hansen(ic, ppl, pmi, wtmu, nmu, fc):
    '''
    Subroutine to normalise the phase function using the method described in Hansen (1971,J.ATM.SCI., V28, 1400)

    PPL,PMI are the forward and backward parts of the azimuthally-integrated
    phase function. The normalization of the true phase fcn is:
    integral over sphere [ P(mu,mu',phi) * dO] = 1
    where dO os the element of solid angle and phi is the azimuthal angle.
    '''
    
    pi = np.pi
    x1 = 2.0 * pi
    if ic == 0:
        rsum = np.zeros(nmu, dtype=float)
        for j in range(nmu):
            rsum[j] = np.sum(pmi[:, j] * wtmu) * x1
        niter = 0
        tsum = np.zeros(nmu)
        
        for niter in range(10000):
            for j in range(nmu):
                tsum[j] = np.sum(ppl[:, j] * wtmu * fc[:, j]) * x1
            testj = np.abs(rsum + tsum - 1.0)
            test = np.max(testj)
            if test < 1e-14:
                break
            for j in range(nmu):
                xj = (1.0 - rsum[j]) / tsum[j]
                for i in range(j+1):
                    xi = (1.0 - rsum[i]) / tsum[i]
                    fc[i, j] = 0.5 * (fc[i, j] * xj + fc[j, i] * xi)
                    fc[j, i] = fc[i, j]
    ppl *= fc
    return (ppl, fc)

@njit(fastmath=True)
def calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, norm, icont, ncont, nphi, fc, pfunc, xmu):
    
    #Integrating the phase function
    pplpl, pplmi = phasint2(nphi, ic, nmu, mu, iscat, cons8, ncons, icont, ncont, pfunc, xmu)

    if norm == 1:
        pplpl, fc = hansen(ic, pplpl, pplmi, wtmu, nmu, fc)

    return (pplpl, pplmi, fc)
            
@njit(fastmath=True)
def matmul(A, B):
    '''
    Matrix multiplication
    '''
    # Get the dimensions of the matrices
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Ensure the matrices are compatible for multiplication
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to number of rows in B")

    # Initialize the result matrix with zeros
    result = np.zeros((rows_A, cols_B))

    # Perform matrix multiplication using nested loops
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]

    return result

@njit
def frob(r):
    return np.sqrt((r*r).sum())

@njit(fastmath=True)
def add(r1, t1, j1, e, nmu,ic):

    bcom = matmul(r1,r1)
    if frob(r1)>0.1: # Frobenius norm < 0.1 for approximation to keep error under 1e-4
        acom = np.linalg.inv(e-bcom)
    else: 
        acom = e + bcom
    ccom = matmul(t1,acom)
    rans = matmul(ccom,r1)
    acom = matmul(rans,t1)
    rans = r1 + acom
    tans = matmul(ccom,t1)

    if ic==0:
        jcom = matmul(r1,j1)
        jcom = jcom + j1
        jans = matmul(ccom,jcom)
        jans = jans + j1
    else:
        jans = j1

    return rans, tans, jans


@njit(fastmath=True)
def numba_diagonal(arr):
    rows, cols = arr.shape
    length = min(rows, cols)
    diagonal = np.empty(length, arr.dtype)
    for i in range(length):
        diagonal[i] = arr[i, i]
    return diagonal

@njit(fastmath=True)
def numba_fill_diagonal(arr, vec):
    for i in range(len(vec)):
        arr[i, i] = vec[i]
        
@njit(fastmath = True)
def numba_sum_diagonal(arr):
    sum_ = 0
    for i in range(arr.shape[0]):
        sum_ += arr[i,i]
    return sum_

@njit(fastmath = True, error_model='numpy')
def double1(ic,nmu,cc,pplpl,pplmi,omega,taut,bc,mminv,e,raman = False): 
    
    ipow0 = 12

    con = omega * np.pi
    del01 = 0.0
    if ic == 0:
        del01 = 1.0
    
    con *= (1.0 + del01)
    
    #Computation of Gamma++ (Plass et al. 1973)
    #GPLPL = MMINV*(E - CON*PPLPL*CC)
    acom = matmul(pplpl,cc)*con
    bcom = e - acom
    gplpl = matmul(mminv,bcom)
    
    #Computation of Gamma+- (Plass et al. 1973)
    #GPLMI = MMINV*CON*PPLMI*CC
    acom = matmul(pplmi,cc)*con
    gplmi = matmul(mminv,acom)
    
    nn = int(np.log2(taut) + ipow0)
    xfac = 1.0 / (2.0 ** nn) if nn >= 1 else 1.0
    tau0 = taut * xfac
    
    # Computation of R, T and J for initial layer
    t1 = e - tau0 * gplpl
    r1 = tau0 * matmul(e,gplmi)
    
    j1 = np.zeros((nmu,1))
    if ic == 0:
        for i in range(nmu):
            j1[i] = (1.0 - omega) * bc * tau0 * mminv[i,i]

    if nn < 1:
        return r1, t1, j1
    
    for n in range(nn):
        r1, t1, j1 = add(r1, t1, j1, e, nmu, ic)
   
    return r1, t1, j1

@njit(fastmath = True)
def idown(ra,ta,ja,rb,tb,jb,u0pl,utmi,e):
    '''
    Calculate the downward radiance at any interior point tau_1 in the layer from tau_0 and tau_2
    
    This function calculates equation 6 from Plass et al. (1973)
    
    I1+ = (E-R10*R12)^-1 * (T01*I0+ + R10*T21*I2- + J01+ + R10*J21- )
    
    Inputs
    ______
    
    ra(nmu,nmu) :: Reflection matrix R10 
    ta(nmu,nmu) :: Transmission matrix T01
    ja(nmu,nmu) :: Source matrix J01+
    rb(nmu,nmu) :: Reflection matrix R12
    tb(nmu,nmu) :: Transmission matrix T21
    jb(nmu,nmu) :: Source matrix J21-
    u0pl(nmu,1) :: Downward intensity I0+
    utmi(nmu,1) :: Upward intensity I2-
    e(nmu,nmu) :: Identity matrix
    
    Outputs
    _______
    
    upl(nmu,1) :: Downward intensity I1+
    '''
    
    #Calculate R10*R12
    acom = matmul(ra,rb)
    
    #Calculate (E-R10*R12)^-1
    bcom = np.linalg.inv(e-acom)
    
    #Calculate T01*I0+
    xcom = matmul(ta,u0pl)
    
    #Calculate R10*T21
    acom = matmul(ra,tb)

    #Calculate R10*T21*I2-
    ycom = matmul(acom,utmi)
    
    #Add previous two results (T01*I0+ + R10*T21*I2-)
    xcom += ycom

    #Calculate R10*J21-
    ycom = matmul(ra,jb)
    
    #Add to total (T01*I0+ + R10*T21*I2- + R10*J21- + J01+)
    xcom += + ycom + ja

    #Multiply by (E-R10*R12)^-1 to get final I1+
    upl = matmul(bcom,xcom)

    return upl

@njit(fastmath = True)
def iup(ra,ta,ja,rb,tb,jb,u0pl,utmi,e):
    '''
    Calculate the upward radiance at any interior point tau_1 in the layer from tau_0 and tau_2
    
    This function calculates equation 5 from Plass et al. (1973)
    
    I1- = (E-R12*R10)^-1 * (T21*I2- + R12*T01*I0+ + J21- + R12*J01+ )
    
    Inputs
    ______
    
    ra(nmu,nmu) :: Reflection matrix R10 
    ta(nmu,nmu) :: Transmission matrix T01
    ja(nmu,nmu) :: Source matrix J01+
    rb(nmu,nmu) :: Reflection matrix R12
    tb(nmu,nmu) :: Transmission matrix T21
    jb(nmu,nmu) :: Source matrix J21-
    u0pl(nmu,1) :: Downward intensity I0+
    utmi(nmu,1) :: Upward intensity I2-
    e(nmu,nmu) :: Identity matrix
    
    Outputs
    _______
    
    umi(nmu,1) :: Upward intensity I1-
    '''
    
    #Calculate R12*R10
    acom = matmul(rb,ra)
    
    #Calculate (E-R12*R10)^-1
    bcom = np.linalg.inv(e-acom)
    
    #Calculate T21*I2-
    xcom = matmul(tb,utmi)
    
    #Calculate R12*T01
    acom = matmul(rb,ta)

    #Calculate R12*T01*I0+
    ycom = matmul(acom,u0pl)
    
    #Add previous two results (T12*I2- + R12*T01*I0+)
    xcom += ycom

    #Calculate R21*J01+
    ycom = matmul(rb,ja)
    
    #Add to total (T12*I2- + R12*T01*I0+ + R21*J01+ + J21-)
    xcom += + ycom + jb

    #Multiply by (E-R12*R10)^-1 to get final I1-
    umi = matmul(bcom,xcom)

    return umi
    


@njit(fastmath = True)
def addp(r1, t1, j1, iscat1,e, rsub, tsub, jsub, nmu): 

    '''
    Calculate the the R,T,J matrices from the combination of two layers
    '''

    if iscat1 == 1:
        rsq = np.zeros_like(r1)
        bcom = np.zeros_like(r1)
        ccom = np.zeros_like(r1)
        rans = np.zeros_like(r1)
        tans = np.zeros_like(r1)
        jcom = np.zeros_like(j1)
        jans = np.zeros_like(j1)
        
        
        # Second layer is scattering
        rsq = matmul(rsub,r1)
        if frob(rsq)>0.01:
            acom = np.linalg.inv(e-rsq)
        else:
            acom = e+rsq
        ccom = matmul(t1,acom)
        rans = matmul(ccom,rsub)
        bcom = matmul(rans,t1)
        rans = r1 + bcom
        tans = matmul(ccom,tsub)
        jcom = matmul(rsub,j1)
        jcom += jsub
        jans = matmul(ccom,jcom)
        jans += j1
        
    else:
        
        # Second layer is non-scattering
        jcom = np.zeros_like(j1)
        tans = np.zeros((nmu,nmu))
        rans = np.zeros((nmu,nmu))
        jans = np.zeros((nmu,1))
        
        jcom = matmul(rsub,j1)
        jcom += jsub
        
        for i in range(nmu):
            ta = t1[i, i]
            for j in range(nmu):
                tb = t1[j, j]   
                tans[i, j] = tsub[i, j] * ta
                rans[i, j] = rsub[i, j] * ta * tb
            jans[i, 0] = j1[i, 0] + ta * jcom[i, 0]
            
    return rans, tans, jans

@njit(fastmath = True)
def angle_quadrature(solar,sol_ang,emiss_ang,mu,nmu):

    if sol_ang > 90.0:
        zmu0 = np.cos(np.radians(180 - sol_ang))
        solar1 = solar*0.0
    else:
        zmu0 = np.cos(np.radians(sol_ang))
        solar1 = solar

    zmu = np.cos(np.radians(emiss_ang))

    isol = 0
    for j in range(nmu-1):
        if zmu0 <= mu[j] and zmu0 > mu[j+1]:
            isol = j
    if zmu0 <= mu[nmu-1]:
        isol = nmu - 2

    iemm = 0
    for j in range(nmu-1):
        if zmu <= mu[j] and zmu > mu[j+1]:
            iemm = j
    if zmu <= mu[nmu-1]:
        iemm = nmu - 2

    u = (mu[isol] - zmu0) / (mu[isol] - mu[isol+1])
    t = (mu[iemm] - zmu) / (mu[iemm] - mu[iemm+1])
    
    return solar1, isol, iemm, t, u 

@njit(fastmath = True, error_model='numpy')
def calc_rtj_matrix(ic,mu,wtmu,bb,tautot,tauscat,tauray,frac,ppln,pmin,pplr,pmir,cc,mminv,e):
    '''
    Calculate the Reflection, Transmission and Source matrices for a given atmospheric layer
    
    Inputs
    ______
    
    mu(nmu) :: Quadrature angles
    wtmu(nmu) :: Weights of the quadrature angles
    bb :: Black-body radiation for the layer 
    tautot :: Total optical depth of the layer (absorption + scattering)
    omega :: Aerosol scattering optical depth of the layer
    tauray :: Rayleigh scattering optical depth of the layer
    frac(ndust) :: Fraction of aerosol scattering by each aerosol population
    ppln(ndust,nmu,nmu) :: Phase matrix in + direction of each aerosol population
    pmin(ndust,nmu,nmu) :: Phase matrix in - direction of each aerosol population
    pplr(nmu,nmu) :: Phase matrix in + direction for Rayleigh scattering
    pmir(nmu,nmu) :: Phase matrix in - direction for Rayleigh scattering
    
    Outputs
    _______
    
    rl(nmu,nmu) :: Reflection matrix
    tl(nmu,nmu) :: Transmission matrix
    jl(nmu,1) :: Source matrix
    iscl :: Flag indicating whether layer is scattering (1) or non-scattering (0)
    '''

    nmu = len(mu)       #Number of quadrature angles
    ncont = len(frac)   #Number of aerosol populations
    omega = (tauscat+tauray)/tautot  #Single scattering albedo of the layer
    
    #Initialising arrays
    rl = np.zeros((nmu,nmu))
    tl = np.zeros((nmu,nmu))
    jl = np.zeros((nmu,1))
    
    #Calculating matrices
    if cc is None:
        cc = np.zeros((nmu,nmu))
        numba_fill_diagonal(cc, wtmu)
    if e is None:
        e = np.identity(nmu)
    if mminv is None:
        mminv = np.zeros((nmu,nmu))
        numba_fill_diagonal(mminv,1./mu)
    
    #Calculating matrices
    if tautot == 0:     #If there is no atmospheric opacity
        iscl = 0
        rl[:,:] *= 0.0
        tl[:,:] *= 0.0
        jl[:,:] *= 0.0
        for i in range(nmu):
            tl[i,i] = 1.0

    elif omega == 0:  #If layer is not scattering
        iscl = 0
        rl[:,:] *= 0.0
        tl[:,:] *= 0.0
        for i1 in range(nmu):
            tex = -mminv[i1,i1]*tautot
            if tex > -200.0:
                tl[i1,i1] = np.exp(tex)
            else:
                tl[i1,i1] = 0.0

            jl[i1,0] = bb*(1.0 - tl[i1,i1])

    else:  #If layer is scattering
        pplpl = (tauray/(tauscat+tauray))*pplr[:,:]
        pplmi = (tauray/(tauscat+tauray))*pmir[:,:]

        for j1 in range(ncont):
            pplpl += tauscat/(tauscat+tauray)*ppln[j1,:,:]*frac[j1] 
            pplmi += tauscat/(tauscat+tauray)*pmin[j1,:,:]*frac[j1] 

        iscl = 1
        rl[:,:], tl[:,:], jl[:,:] = double1(ic,nmu,cc,pplpl,pplmi,omega,tautot,bb,mminv,e,raman = False)
    
    return rl,tl,jl,iscl



@njit(fastmath = True,parallel=False, cache = True, error_model='numpy')
def scloud11wave_core(phasarr, radg, sol_angs, emiss_angs, solar, aphis, lowbc, brdf_matrix, mu1, wt1, nf,
                vwaves, bnu, taus, tauray,omegas_s, nphi,iray,imie, lfrac):
    
    """
    Calculate spectrum using the adding-doubling method.

    Parameters
    ----------
    phasarr(NMODES,NWAVE,2,NPHAS):
        Contains phase functions (phasarr[:,:,0,:]) and corresponding angle grids (phasarr[:,:,1,:])
    radg(NWAVE,NMU): 
        Incident intensity at the bottom of the atm
    sol_ang(NPATH):
        Solar zenith angle (degrees)
    emiss_angs(NPATH):
        Emission zenith angle (degrees)
    solar(NWAVE):        
        Incident solar flux at the top of the atm
    aphi(NPATH):
        Azimuth angle between Sun and observer
    lowbc:
        Lower boundary condition: 0 = thermal (no surface), 1 = Lambert reflection (surface)
    brdf_matrix(NWAVE,NMU,NMU,NF+1):
        Decomposed BRDF matrix of the surface
    mu1(NMU):  
        Zenith angle point quadrature
    wt1(NMU):  
        Zenith angle point quadrature
    nf:
        Maximum number of terms in azimuth Fourier expansion 
    vwaves(NWAVE):
        Input wavelengths.
    bnu(NWAVE,NLAY):
        Mean Planck function in each layer
    tau(NWAVE,NG,NLAY):
        Total optical thickness of each layer
    tauray(NWAVE,NLAY):
        Rayleigh optical thickness of each layer
    omegas(NWAVE,NG,NLAY):
        Single scattering albedo of each layer (aerosol + Rayleigh scattering)
    nphi:
        Number of azimuth integration ordinates
    iray:
        Flag for using rayleigh scattering
    iray:
        Flag for phase function type
    lfrac(NWAVE,NCONT,NLAY):
        Fraction of scattering contributed by each type in each layer
    
    Returns
    -------
    rad(NWAVE):
        Upwards radiance at each wavelength
    """
    
    nwave = len(vwaves)
    nmu = len(mu1)
    ngeom = len(emiss_angs)
    ng = taus.shape[1]
    nlay = taus.shape[2]
    ncont = phasarr.shape[0]
    pi = np.pi

    ltot = nlay
    if lowbc > 0:
        ltot += 1

    xfac = 0.
    xfac = np.sum(mu1*wt1)
    xfac = 0.5/xfac    

    # Reset the order of angles
    mu = mu1[::-1]
    wtmu = wt1[::-1]
    
    # Setting up radiance matrices
    rad = np.zeros((ngeom,ng,nwave))

    yx = np.zeros((4))
    
    u0pl = np.zeros((nmu,1))           #Incident downward radiance (+) at top of the atmosphere (solar)
    utmi = np.zeros((nmu,1))           #Incident upward radiance (-) at bottom of atmosphere or surface (ground)
    umi = np.zeros((nlay,nmu))         #Upward radiance (-)
    upl = np.zeros((nlay,nmu))         #Downward radiance (+)

    ppln = np.zeros((ncont, nmu, nmu)) #Phase matrix of the scattering aerosols in + direction
    pmin = np.zeros((ncont, nmu, nmu)) #Phase matrix of the scattering aerosols in - direction
    pplr = np.zeros((nmu, nmu))        #Phase matrix for Rayleigh scattering in + direction
    pmir = np.zeros((nmu, nmu))        #Phase matrix for Rayleigh scattering in - direction
    
    rl = np.zeros((nmu,nmu))           #Reflection matrix of each layer
    tl = np.zeros((nmu,nmu))           #Transmission matrix of each layer
    jl = np.zeros((nmu,1))             #Source matrix of each layer
    
    rs = np.zeros((nmu,nmu,nf+1))      #Reflection matrix of the surface
    ts = np.zeros((nmu,nmu,nf+1))      #Tranmission matrix of the surface
    js = np.zeros((nmu,1,nf+1))        #Source matrix of the surface

    rcomb = np.zeros((nmu,nmu,nf+1))   #Combined reflection matrix from multiple layers
    tcomb = np.zeros((nmu,nmu,nf+1))   #Combined transmission matrix from multiple layers
    jcomb = np.zeros((nmu,1,nf+1))     #Combined source matrix from multiple layers

    # Setting up matrices of constants 
    e = np.identity(nmu)
    mm = np.zeros((nmu, nmu))
    numba_fill_diagonal(mm, mu[:nmu])
    mminv = np.zeros((nmu, nmu))
    numba_fill_diagonal(mminv, 1./mu[:nmu])
    cc = np.zeros((nmu, nmu))
    numba_fill_diagonal(cc, wtmu)
    ccinv = np.zeros((nmu, nmu))
    numba_fill_diagonal(ccinv, 1/wtmu[:nmu])
    
    radg = radg[:,::-1]

    fc = np.ones((ncont+1,nmu,nmu))
    

    #Performing initial checks on whether all spectra are for looking up or looking down
    if np.all(emiss_angs < 90):
        lookdown = True
    elif np.all(emiss_angs > 90):
        lookdown = False
    else:
        raise ValueError("Emission angles are a mix of values above and below 90 degrees.")

        
    #Main loop: Iterating through each wavelength
    for ig in range(ng):
        # Getting correct layer properties for this g-ordinate
        
        tau = taus[:,ig,:]
        omegas = omegas_s[:,ig,:]
        
        for widx in range(nwave):
            
            # Expand into successive fourier components until convergence or ic = nf
            
            for ic in range(nf+1):
                ppln.fill(0.)
                pmin.fill(0.)
                pplr.fill(0.)
                pmir.fill(0.)

                for j1 in range(ncont):                    
                    pfunc = phasarr[j1, widx, 0, :]
                    xmu   = phasarr[j1, widx, 1, :]
                    if imie == 0:
                        iscat = 2
                    else:
                        iscat = 4
                    ncons = 0
                    cons8 = pfunc
                    norm = 1
                    pplpl, pplmi, fc[j1] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                                norm, j1, ncont, nphi, fc[j1], pfunc, xmu)
                    # Transfer matrices to those for each scattering particle
                    ppln[j1,:,:] = pplpl
                    pmin[j1,:,:] = pplmi

                if iray > 0:
                    iscat = 0
                    ncons = 0
                    pplpl, pplmi, fc[ncont] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                                1, ncont, ncont, nphi, fc[ncont], pfunc, xmu)

                    # Transfer matrices to storage
                    pplr[:,:] = pplpl
                    pmir[:,:] = pplmi
                    
                #Computing RTJ matrices for the surface
                surface_defined = False
                if(lowbc > 0):
                    
                    js[:,0,ic] = radg[widx]   #Emissivity is already accounted for in radg
                    for j in range(nmu):
                        rs[:,j,ic] = 2. * (brdf_matrix[widx,:,j,ic] * np.pi) * mu[j] * wtmu[j]
                    rs[:,:,ic]*= xfac
                    ts[:,:,ic].fill(0.0)
    
                    if lookdown == True:
                        surface_defined = True
                        jcomb[:,:,ic] = js[:,:,ic]
                        rcomb[:,:,ic] = rs[:,:,ic]
                        tcomb[:,:,ic] = ts[:,:,ic]

                # Main loop: computing R,T,J for each layer
                for l in range(0,nlay):
                    
                    if lookdown is True:
                        k = l
                    else:
                        k = nlay - 1 - l
                        
                    #Defining the properties of the layer
                    taut = tau[widx,k]
                    bc = bnu[widx,k]
                    omega = omegas[widx,k]
                    if omega < 0:
                        omega = 0.0
                    if omega > 1:
                        omega = 1.0

                    tauscat = taut*omega
                    taur = tauray[widx,k]
                    tauscat = tauscat-taur

                    if tauscat < 0:
                        tauscat = 0.0  
                        
                    omega = (tauscat+taur)/taut
                    frac = lfrac[widx,:,k]

                    #Calculating the matrices
                    rl, tl, jl, iscl = calc_rtj_matrix(ic,mu,wtmu,bc,taut,tauscat,taur,frac,ppln,pmin,pplr,pmir,cc,mminv,e)
                    
                    #Combining layers along the path
                    if l == 0 and surface_defined == False:
                        jcomb[:,:,ic] = jl
                        rcomb[:,:,ic] = rl
                        tcomb[:,:,ic] = tl
                    else:
                        rcomb[:,:,ic], tcomb[:,:,ic], jcomb[:,:,ic] = addp(rl, tl, jl, iscl,
                                                    e, rcomb[:,:,ic], tcomb[:,:,ic], jcomb[:,:,ic], nmu)

                    #print('rcomb',rcomb[:,:,ic])
                    #print('tcomb',tcomb[:,:,ic])
                    #print('jcomb',jcomb[:,:,ic])
                    #raise ValueError('hola')

                if ic != 0:
                    jcomb[:,:,ic].fill(0.0)

            #calculating the spectra
            for ipath in range(ngeom):
            
                converged = False
                conv1 = False
                defconv = 1e-5
                
                sol_ang = sol_angs[ipath]
                emiss_ang = emiss_angs[ipath]
                aphi = aphis[ipath]
                
                if emiss_ang < 90.: #Nadir-viewing geometry
                    new_emi = emiss_ang
                elif emiss_ang > 90.: #Upward-looking geometry
                    new_emi = 180. - emiss_ang
                    
                solar1, isol, iemm, t, u = angle_quadrature(solar,sol_ang,new_emi,mu,nmu)
            
                for ic in range(nf+1):
                    
                    #Calculating the spectrum
                    for j in range(nmu):
                        u0pl[j] = 0.0
                        if ic == 0:
                            utmi[j] = radg[widx,j]
                        else:
                            utmi[j] = 0.0

                    ico = 0

                    if lookdown is True:
                        for imu0 in range(isol, isol+2): 
                            u0pl[imu0] = solar1[widx] / (2.0 * np.pi * wtmu[imu0])
                            acom = matmul(rcomb[:,:,ic],u0pl)
                            bcom = matmul(tcomb[:,:,ic],utmi)
                            acom += bcom
                            umi = acom + jcomb[:,:,ic]
                            for imu in range(iemm, iemm+2): 
                                yx[ico] = umi[imu,0]
                                ico += 1
                                u0pl[imu0] = 0.0 
                    else:
                        for imu0 in range(isol, isol+2): 
                            
                            u0pl[imu0] = solar1[widx] / (2.0 * np.pi * wtmu[imu0])
                            if lowbc == 0: #We are at the very bottom of the atmosphere+surface system
                                acom = np.ascontiguousarray(tcomb[:,:,ic]) @ u0pl
                                bcom = np.ascontiguousarray(rcomb[:,:,ic]) @ utmi
                                acom += bcom
                                upl = acom + jcomb[:,:,ic]
                            else: #We are not at the very bottom, so we need to compute the internal radiation field
                                upl = idown(rcomb[:,:,ic],tcomb[:,:,ic],jcomb[:,:,ic],rs[:,:,ic],ts[:,:,ic],js[:,:,ic],u0pl,utmi,e)
                            
                            for imu in range(iemm, iemm+2): 
                                yx[ico] = upl[imu,0]
                                ico += 1
                                u0pl[imu0] = 0.0 

                    drad = ((1-t)*(1-u)*yx[0] + t*(1-u)*yx[1] + t*u*yx[3] + (1-t)*u*yx[2]) * np.cos(ic*aphi * np.pi / 180.0)

                    if ic > 0:
                        drad *= 2

                    rad[ipath,ig,widx] += drad
                    conv = np.abs(drad / rad[ipath,ig,widx])
                    
                    # If sure about convergence, exit loop over fourier components
                    if conv < defconv and conv1:
                        break

                    if conv < defconv:
                        conv1 = True
                    else:
                        conv1 = False
    
    return rad



