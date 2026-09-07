# Estimating a biodegradation rate constant from groundwater monitoring data

*Literature survey and method comparison, September 2026. Companion to the estimand table in
`README.md` and to the three estimators in `biodeg_rates/`.*

This note answers two questions that were put to us directly, surveys the method families in the
published literature, and records where our three implemented methods sit among them. The
literature search was run through Consensus across thirteen queries covering analytical plume
inversions, isotope methods, mass-flux methods, in-situ tests, direct assays, source-zone
depletion, censored-data statistics, and Bayesian parameter estimation.

## 1. Can a rate constant be estimated from a single well's time series?

Yes, subject to one condition that governs everything downstream: the resulting number is not a
biodegradation rate.

A concentration-versus-time slope at one well lumps destruction together with dilution,
dispersion, sorption and source decline. The literature has been explicit about this since the
EPA issue paper that named the distinction between concentration-versus-time and
concentration-versus-distance rate constants and warned that they represent very different
attenuation processes [2].

The defensible position is therefore threefold: (i) the time series yields an **apparent
attenuation rate**, (ii) that quantity is the correct one for plume-stability and
remediation-timeframe questions, and (iii) isolating the biological term requires either removing
dilution analytically or measuring destruction through an independent physical signal.

## 2. How is uncertainty assigned?

Uncertainty in these estimates has four layers. Most reporting covers only the first.

**Layer 1, sampling noise on the fitted slope.** The bootstrap or rank-based confidence interval
on the rate. This is what the standard trend toolkits report [7]. Zhou et al. quantify the
limitation: the Theil-Sen slope is accurate but imprecise for minor trends, and the power of the
Mann-Kendall test degrades interactively with missing data and averaging window size [8].

**Layer 2, censoring.** Non-detects are not zeros, and substitution at RL/2 is a convention rather
than an estimator. Kaplan-Meier, robust order statistics (ROS) and maximum-likelihood estimation
for left-censored data outperform substitution, with ROS holding to roughly 50 percent censoring
and MLE to roughly 25 percent [33]. A mixed-model formulation handles monitoring years that are
entirely non-detect [34]. Maximum-likelihood treatment of left-censored data changes trend and
risk conclusions materially at high censoring fractions [32].

**Layer 3, structural and parameter uncertainty.** This usually dominates and is usually omitted.
Stenback found that varying transverse and longitudinal dispersivity across one order of magnitude
moved the estimated rate constant nearly threefold [4]. Beyer's synthetic-truth experiment is the
number to quote: estimators overstated the known rate by an average factor of about 2 for wide
plumes and about 5 for narrow ones, with individual values ranging from 0.5 to 20 [5].

**Layer 4, predictive uncertainty.** A different quantity again, and the one that matters for a
closure schedule. McHugh et al. split each GeoTracker monitoring record in half and found the
first-half attenuation rate correlates with the second-half rate at r of about -0.11 to -0.12,
across benzene, MTBE and TCE. The negative correlation persisted in subsets restricted to good
model fits, to statistically significant first-half trends, and to records with no change in
remedy [24].

Two formal machineries in the literature carry these layers together. Bayesian hierarchical
models separate natural site-to-site variability from experimental error and predict at a new site
with an explicit several-orders-of-magnitude spread. They also show which additional measurements
actually narrow it: mineral composition barely helped, whereas Fe(II) content reduced uncertainty
in the first-order rate constant by nearly two orders of magnitude [22]. Monte Carlo inverse
modelling ties the rate's confidence interval to uncertainty in source geometry and the hydraulic
conductivity field, rather than treating the rate as if the flow field were known [23, 35].

## 3. Method families in the literature

### 3.1 Analytical inversion of a steady-state plume

The family our Method 2 belongs to. Wiedemeier and colleagues published the original pair in 1996:
normalization against a biologically recalcitrant tracer travelling in the same water, and
inversion of a one-dimensional analytical solution for a plume assumed to be at dynamic steady
state [1]. Buscheck and Alcantar's centreline regression (1995, Battelle Press) became the
default. Zhang's Domenico normalization corrects its bias, measured at 21 percent for a
two-dimensional plume and 65 percent for a three-dimensional one on Buscheck and Alcantar's own
Fairfax Terminal data [3]. Stenback's two-dimensional least-squares fit uses all downgradient
wells and requires no centreline, returning rates as low as half the one-dimensional values [4].
Yuan proposes a further stepwise improvement [6]. Beyer's synthetic-plume study evaluates the
whole family against known truth [5].

### 3.2 Compound-specific isotope analysis (CSIA)

The one family that isolates destruction on physical rather than modelling grounds: isotope
fractionation occurs when bonds break, so the Rayleigh equation converts an observed isotope shift
into an extent of degradation [9]. Bashir derives distance- and time-dependent first-order rate
constants directly, with maxima of 11 x 10^-3 per day for beta-HCH [11]. De Vera reports extents
of degradation of 47 to 99 percent for 1,2-DCB and 21 to 73 percent for 1,2,4-TCB [12]. The
essential caution is van Breukelen's open-system correction: dilution contributed several to many
times more than biodegradation at a benzene plume, and field-derived enrichment factors
underestimate the true values as a consequence [10].

**This is the strongest available answer to a regulator who asks for proof of destruction rather
than dilution, and we do not currently do it.**

### 3.3 Mass flux and mass discharge

Convert the concentration field into a mass balance between two control planes; the difference
gives an attenuation rate. Bockelmann's integral pumping approach at a former gasworks produced
first-order rates of 1.4 x 10^-2 to 1.3 x 10^-1 per day for BTEX and 3.7 x 10^-4 to 3.1 x 10^-2
per day for PAH [13], with a later field-scale application giving effective rates of 4.7 x 10^-7
to 1.6 x 10^-6 per second [14]. ProfileFlux estimates mass discharge from high-resolution vertical
profiles where a transect is impractical, validated at 117 to 170 g/year against 143 g/year from
the traditional transect method [15].

### 3.4 In-situ tests

The single-well push-pull test injects a tracer plus reactive solutes, extracts the mixture from
the same well, and computes reaction rates from the breakthrough curves [16]. It measures
metabolism in the aquifer over days rather than years, and has been applied to petroleum
hydrocarbons in fractured bedrock [17].

### 3.5 Direct assays and molecular tools

Carbon-14 labelled assays quantify rates far below what concentration trends can resolve: 0.0021
per year for 1,4-dioxane, a half-life beyond 300 years [20], and 0.0092 to 0.24 per year for
chlorinated ethenes in aquifer materials [19] and intact rock cores [21]. Molecular biological
tools estimate an apparent first-order constant from biomarker gene abundance, for example bssA
for anaerobic toluene degradation, and can establish whether the microbial community has
acclimated at all [18].

### 3.6 Source-zone and NSZD methods

These answer a different question: how fast the source depletes, not how fast the plume degrades.
Benzene decay in the source zone ran roughly six times slower than plume rates over nearly 18
years of controlled release monitoring, with decay rates of 0.45 to 1.75 per year [27].
Site-average NSZD rates across 40 LNAPL sites span 650 to 152,000 L/ha/yr with a median of 9,540,
and different measurement methods applied at the same site differ by a median factor of 2.1 [28].
Soil-gas gradient methods offer a simplified screening route [29], and LNAPL composition drives
which depletion mechanism dominates [30].

### 3.7 Portfolio and space-time statistical methods

Where our Methods 1 and 3 sit. Non-parametric trend toolkits for plume stability [7], multi-site
plume studies establishing population distributions of length, stability and attenuation rate
[25, 26], and space-time geostatistical models of the concentration surface [31], which is the
family our tensor-product P-spline belongs to.

## 4. Where our three methods sit, and the gap

| Our method | Family | Estimand | Proves destruction? |
|---|---|---|---|
| 1, Mann-Kendall + Theil-Sen | 3.7 | apparent point decay | no |
| 2, Domenico-normalized | 3.1 | flowpath lambda, dilution removed | by inference from a transport model |
| 3, spatio-temporal P-spline | 3.7 | apparent centre decay | no |
| 3, same surface, integrated | 3.7 | plume mass decay, -d ln M(t)/dt | no; spreading inside the footprint is removed, boundary export is not |

The mass estimand is the closest of the four to the mass-balance logic of the flux methods in
3.3, but it is a mass balance over a footprint whose boundary we draw rather than across a
measured control plane, so it inherits neither the flux methods' rigour nor their cost. Its
measured bias and boundary sensitivity are recorded in `README.md`.

All four are inversions of routine monitoring data. Every method that independently *proves*
destruction (CSIA, push-pull, carbon-14, biomarkers) requires field or laboratory work we do not
currently perform. If the binding regulatory question is whether a plume is degrading or merely
spreading, CSIA is the strongest available answer and the clearest candidate for extending this
work.

Note also that microcosm-derived rate parameters can reproduce observed behaviour in a
heterogeneous system, but subsurface heterogeneity strongly influences the accuracy of those
predictions: homogeneous-domain simulations underpredicted the extent of dechlorination by an
average of 13 percent, with a maximum discrepancy of 45 percent [36].

## 5. Historical account

The first field methods arrived with the intrinsic bioremediation programs of the mid-1990s, when
regulators began accepting natural attenuation as a remedy and asked practitioners to defend it
with a number. Wiedemeier and colleagues published two approaches in 1996, both built from
ordinary monitoring wells [1]. The first normalizes BTEX against a recalcitrant tracer moving in
the same water, so that whatever dilutes the tracer is subtracted from the contaminant and the
excess loss is credited to biodegradation. The second assumes the plume has reached a dynamic
steady state, then inverts a one-dimensional analytical solution to recover the decay that would
hold a plume at the length observed in the field. Buscheck and Alcantar had proposed a centreline
regression the year before, and it became the default because it asked for so little:
concentrations, a gradient, and a straight line drawn down the plume axis. Thus a consultant in
1997 could produce a defensible-looking rate constant in an afternoon.

The decade that followed was spent finding out what that afternoon's number measured. Zhang showed
in 2003 that the centreline method overstates the rate when the source is small relative to
lateral spreading, by 21 percent for a two-dimensional plume and 65 percent for a
three-dimensional one, using Buscheck and Alcantar's own Fairfax Terminal data [3]. Stenback
fitted all downgradient wells rather than the centreline in 2004, recovered rates as low as half
the one-dimensional values, and found that moving dispersivity across one order of magnitude moved
the estimate threefold [4]. Beyer then ran the experiment that settles the matter, generating
synthetic plumes with a known truth and investigating them with simulated well networks: the
estimators overstated the true rate by an average factor of about 2 for wide plumes and about 5
for narrow ones, with single values ranging from 0.5 to 20 [5]. We were measuring the plume and
calling it the bacteria. The distinction pays for itself, because a benzene rate overstated
fivefold turns a plume that needs about 33 years to fall two orders of magnitude into one that
appears to need seven, and a closure schedule built on the second number will fail in front of a
regulator. Newell and colleagues named the problem for the EPA in 2003, separating
concentration-versus-time from concentration-versus-distance and warning that the two answer
different questions [2].

Since then the field has moved in two directions at once. The first is physical measurement that
does not depend on inverting a transport model: (i) compound-specific isotope analysis, where
preferential consumption of the light isotope leaves a Rayleigh signature that only bond breaking
can produce [9], (ii) mass-flux transects across control planes, which turn a concentration field
into a mass balance [13], (iii) single-well push-pull tests, which measure metabolism in the
aquifer over days rather than years [16], and (iv) carbon-14 labelled assays and qPCR biomarkers,
which reach rates as slow as 0.002 per year and can tell an acclimated microbial community from a
hopeful one [18, 20]. The second direction is statistical, and it is less comfortable. McHugh and
colleagues went to the California GeoTracker database in 2023, split each monitoring record in
half, and found the first-half attenuation rate correlates with the second half at r of about
-0.11, which is to say a site's own history does not predict its own future [24]. The same
humility applies to our own work, where the dilution-removed flowpath estimator carries a mean
absolute error of 183 percent against synthetic truth in its one-dimensional form and improves to
62 percent only when the transverse structure is fitted rather than assumed. In so doing we give a
regulator what the data supports, which is a rate with its estimand named and its uncertainty
attached, rather than a single tidy number that would describe no site.

## 6. Accuracy caveats on this document

**Buscheck and Alcantar (1995) is not in the reference list.** It is a Battelle Press conference
chapter (in Hinchee, Wilson and Downey, eds., *Intrinsic Bioremediation*, pp. 109-116), not a
journal article, so it is not indexed by the search used here. Every statement about it in this
document is second-hand through Zhang [3] and Stenback [4]. Obtain the primary reference before
citing it in a regulatory submission.

**The population-median anchor is our assumption, not McHugh's finding (corrected in the code).**
The paper establishes that a site's own history is a poor predictor of its own future rate, with
r of about -0.11 [24]. It does not test the portfolio median as a predictor, so the step from
"site history does not predict" to "use the population median instead" is our inference. It is
the load-bearing assumption under the informed-prior combine: if it is wrong, the combine is
anchored to the wrong quantity. Earlier versions of `biodeg_rates/prior.py` attributed this step
to the paper; the module docstring and the prior's `note` field now separate what the paper shows
from what we assume. The plausibility check that flags a site estimate outside the population band
is likewise our rule, not the paper's.

**The McHugh volume, issue and page range** recorded in `CITATION.cff` and `prior.py` come from
our own records, not from the search result. The title and journal have been verified.

## References

[1] [Approximation of Biodegradation Rate Constants for Monoaromatic Hydrocarbons (BTEX) in Ground Water](https://consensus.app/papers/details/f9d0015241555494aacc9469ee841a62/?utm_source=claude_code) (T. Wiedemeier et al., 1996, Ground Water Monitoring and Remediation, 122 citations)

[2] [Calculation and Use of First-Order Rate Constants for Monitored Natural Attenuation Studies](https://consensus.app/papers/details/01c90c6819ae5eb097d88afca66646a6/?utm_source=claude_code) (C. Newell et al., 2003, US EPA Issue Paper, 75 citations)

[3] [An Improved Method for Estimation of Biodegradation Rate with Field Data](https://consensus.app/papers/details/71484c3c0fc1507d9e75acf9c187368e/?utm_source=claude_code) (You-Kuan Zhang et al., 2003, Groundwater Monitoring & Remediation, 14 citations)

[4] [Impact of transverse and longitudinal dispersion on first-order degradation rate constant estimation](https://consensus.app/papers/details/ae529f8dbf5d5609a34174cb289c3a7b/?utm_source=claude_code) (G. Stenback et al., 2004, Journal of Contaminant Hydrology, 27 citations)

[5] [Determination of First-Order Degradation Rate Constants from Monitoring Networks](https://consensus.app/papers/details/65af371f4ec952a79c31c1d9607f0c95/?utm_source=claude_code) (C. Beyer et al., 2007, Groundwater, 25 citations)

[6] [An improved analytical approach to estimate in situ biodegradation rates](https://consensus.app/papers/details/01685f2e8df85cf1bf3d48a42866ec78/?utm_source=claude_code) (Liu Yuan et al., 2008)

[7] [GSI Mann-Kendall Toolkit for Quantitative Analysis of Plume Concentration Trends](https://consensus.app/papers/details/7c97f671777051f7a55ac7084e81aba3/?utm_source=claude_code) (J. Connor et al., 2014, Groundwater, 11 citations)

[8] [Do the Mann-Kendall test and Theil-Sen slope fail to inform trend significance and magnitude in hydrology?](https://consensus.app/papers/details/d14a9c0a592e525a9e7b7e12b9b47124/?utm_source=claude_code) (Jiahua Zhou et al., 2023, Hydrological Sciences Journal, 37 citations)

[9] [Quantification of organic pollutant degradation in contaminated aquifers using compound specific stable isotope analysis - Review of recent developments](https://consensus.app/papers/details/0bb551049abe58c69705b8b33ff0fa76/?utm_source=claude_code) (M. Thullner et al., 2011, Organic Geochemistry, 180 citations)

[10] [Quantifying the degradation and dilution contribution to natural attenuation of contaminants by means of an open system Rayleigh equation](https://consensus.app/papers/details/f144087d2f405aee8a2b90be68e154bc/?utm_source=claude_code) (B. V. van Breukelen, 2007, Environmental Science & Technology, 19 citations)

[11] [Evaluating degradation of hexachlorocyclohexane (HCH) isomers within a contaminated aquifer using compound-specific stable carbon isotope analysis (CSIA)](https://consensus.app/papers/details/7b842e2f32c85ef487e9dc7187c09a2a/?utm_source=claude_code) (S. Bashir et al., 2015, Water Research, 57 citations)

[12] [Compound-specific isotope analysis (CSIA) evaluation of degradation of chlorinated benzenes (CBs) and benzene in a contaminated aquifer](https://consensus.app/papers/details/b4131831091f5f3ab3b5ac8c7c30b4ba/?utm_source=claude_code) (J. De Vera et al., 2022, Journal of Contaminant Hydrology, 17 citations)

[13] [An analytical quantification of mass fluxes and natural attenuation rate constants at a former gasworks site](https://consensus.app/papers/details/08d7cf600ea95024a49d0ced00b3f94c/?utm_source=claude_code) (Alexander Bockelmann et al., 2001, Journal of Contaminant Hydrology, 124 citations)

[14] [Field scale quantification of contaminant mass fluxes and natural attenuation rates using an integral investigation approach](https://consensus.app/papers/details/558add627ffe5c01aa9d4b5da013a9c6/?utm_source=claude_code) (Alexander Bockelmann et al., 2020)

[15] [A novel concept for estimating the contaminant mass discharge of chlorinated ethenes emanating from clay till sites](https://consensus.app/papers/details/92b1ab23060a5ee0b769c608be7e6006/?utm_source=claude_code) (Louise Rosenberg et al., 2022, Journal of Contaminant Hydrology, 14 citations)

[16] [Single-Well, "Push-Pull" Test for In Situ Determination of Microbial Activities](https://consensus.app/papers/details/ff00c8f3b8c554b4a7d466f5d5374d76/?utm_source=claude_code) (J. Istok et al., 1997, Groundwater, 271 citations)

[17] [Estimating In Situ Biodegradation Rates of Petroleum Hydrocarbons and Microbial Population Dynamics by Performing Single-Well Push-Pull Tests in a Fractured Bedrock Aquifer](https://consensus.app/papers/details/c772b74c48715a7fb6c6bfa6b8510aaf/?utm_source=claude_code) (Yunchul Cho et al., 2013, Water, Air, & Soil Pollution, 14 citations)

[18] [Field-Scale qPCR Data to Estimate Rate Constants for Toluene Biodegradation in Groundwater](https://consensus.app/papers/details/bad1c0fedef157bb964f759231c26721/?utm_source=claude_code) (Giovanni Pilloni et al., 2025, Groundwater Monitoring & Remediation, 5 citations)

[19] [Quantification of Degradation Rate Constants in Aquifer Materials Using Carbon-14 Chlorinated Ethenes](https://consensus.app/papers/details/be60e0f811375060bde1ae315fdb73c8/?utm_source=claude_code) (David L. Freedman et al., 2025, Groundwater Monitoring & Remediation, 3 citations)

[20] [Evaluation of natural attenuation of 1,4-dioxane in groundwater using a 14C assay](https://consensus.app/papers/details/13bdf09c0a5753eab2958392d76315ac/?utm_source=claude_code) (A. Garcia et al., 2021, Journal of Hazardous Materials, 10 citations)

[21] [Use of carbon-14 trichloroethene to determine degradation rate constants in rock core microcosms](https://consensus.app/papers/details/4ff33e89c1fb51018602a0205fdbbcc6/?utm_source=claude_code) (Hao Wang et al., 2025, Journal of Contaminant Hydrology, 1 citation)

[22] [Predicting Abiotic TCE Transformation Rate Constants - A Bayesian Hierarchical Approach](https://consensus.app/papers/details/26f2e103a3c256f39784e7b18642d4b3/?utm_source=claude_code) (A. Storiko et al., 2024, Groundwater Monitoring & Remediation, 4 citations)

[23] [Probabilistic modeling of natural attenuation of petroleum hydrocarbons](https://consensus.app/papers/details/a5ec69af57c15a5a9bad619dfdb67100/?utm_source=claude_code) (A. Hosseini, 2009, 6 citations)

[24] [Forecasting Groundwater Remediation Timeframes: Site-Specific Temporal Monitoring Results May Not Predict Future Performance](https://consensus.app/papers/details/0331c19898b05244888165bde1907181/?utm_source=claude_code) (Thomas E. McHugh et al., 2023, Groundwater Monitoring & Remediation, 1 citation)

[25] [Use of Long-Term Monitoring Data to Evaluate Benzene, MTBE, and TBA Plume Behavior in Groundwater at Retail Gasoline Sites](https://consensus.app/papers/details/c95cf6487b2257d69d153a5ef142f70e/?utm_source=claude_code) (R. Kamath et al., 2012, Journal of Environmental Engineering, 24 citations)

[26] [A Comparative Plume Study of DRO, GRO, Benzene, and MTBE: Implications for Risk Management](https://consensus.app/papers/details/c2fd6218198555eb983cbe0b18269221/?utm_source=claude_code) (K. O'Reilly et al., 2021, Groundwater Monitoring & Remediation, 3 citations)

[27] [Long-term evaluation of hydrocarbon degradation rates within the source zone under natural and enhanced attenuation for site closure purposes](https://consensus.app/papers/details/ad667bc8c0705dd098afaa0c69d918f7/?utm_source=claude_code) (M. Schneider et al., 2024, Remediation Journal, 1 citation)

[28] [Natural source zone depletion (NSZD) insights from over 15 years of research and measurements: A multi-site study](https://consensus.app/papers/details/b0c865e81872573c838d48c025a8cd49/?utm_source=claude_code) (Poonam R. Kulkarni et al., 2022, Water Research, 24 citations)

[29] [Soil gas gradient method for estimating natural source zone depletion rates of LNAPL and specific chemicals of concern](https://consensus.app/papers/details/16f8cfb117115f6f9ef9a5b20633fd87/?utm_source=claude_code) (I. Verginelli et al., 2024, Water Research, 13 citations)

[30] [Impacts of LNAPL types on mechanisms and rate of natural source zone depletion](https://consensus.app/papers/details/e9ce066818115d9caf92be8caad10309/?utm_source=claude_code) (Junjie Guan et al., 2024, Environmental Pollution, 8 citations)

[31] [Space-Time Distribution of Trichloroethylene Groundwater Concentrations: Geostatistical Modeling and Visualization](https://consensus.app/papers/details/ecaf532aa7e85513a322fea30876a560/?utm_source=claude_code) (Pierre Goovaerts et al., 2023, Mathematical Geosciences, 3 citations)

[32] [A statistical assessment of micropollutants occurrence, time trend, fate and human health risk using left-censored water quality data](https://consensus.app/papers/details/2a55ba113848533c8047c38f6aac0b02/?utm_source=claude_code) (B. Cantoni et al., 2020, Chemosphere, 22 citations)

[33] [Comparison of methods to assess the accuracy of the incorporation of censored chemical data in descriptive statistical analysis of contaminated groundwater](https://consensus.app/papers/details/9eb4a325c86559ffb69d1aec5f2b3b7b/?utm_source=claude_code) (Vinicius Rodrigues dos Santos et al., 2023, Aguas Subterraneas, 2 citations)

[34] [Trend detection with non-detects in long-term monitoring, a mixed model approach](https://consensus.app/papers/details/c5e3186540905ade9f98b7331b443492/?utm_source=claude_code) (M. Skold, 2023, Environmental Monitoring and Assessment, 2 citations)

[35] [A surrogate-based sensitivity quantification and Bayesian inversion of a regional groundwater flow model](https://consensus.app/papers/details/6bf5669a15e55c5ebfdbe4f2365d23b6/?utm_source=claude_code) (Mingjie Chen et al., 2018, Journal of Hydrology, 38 citations)

[36] [Exploration of processes governing microbial reductive dechlorination in a heterogeneous aquifer flow cell](https://consensus.app/papers/details/8bb07430c34d559cb59897492f0d641a/?utm_source=claude_code) (Lurong Yang et al., 2021, Water Research, 12 citations)

---

*Literature search performed with Consensus (consensus.app), September 2026.*
