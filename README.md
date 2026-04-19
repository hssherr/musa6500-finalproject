# Detecting Structural Distress at Scale: A Geospatial Foundation Model Approach to Urban Building Safety

**Course:** MUSA 6500: Geospatial Machine Learning in Remote Sensing  
**Instructors:** Nissim Lebovits, Guray Erus  
**Authors:** Jason Fan, Henry Sywulak-Herr

---

## Problem Definition & Use Case

Philadelphia's aging row home stock presents a unique structural challenge: a single neglected roof can compromise the stability of an entire block. With over 40,000 vacant lots, buildings classified as Imminently Dangerous (ID) within the Philadelphia Property Maintenance Code represent the most severe category of code violation, where a structure is at "imminent danger of failure or collapse."

While the Department of Licenses and Inspections (L&I) maintains a database of vacant property violations—including the Vacant Property Indicator (VPI), which has recently resumed operations after an extended hiatus, though data quality remains inconsistent—administrative data is often reactive and lagged. Mayor Cherelle Parker's administration has also indefinitely paused systematic data collection on vacant lots, leaving gaps in real-time intelligence on structures that pose the greatest immediate threat to Philadelphia residents.

The target user for this project is L&I's Emergency Services Unit and the Philadelphia Land Bank, though it would be broadly useful to most infrastructure- and service-related organizations in Philadelphia (PECO, USPS, Philadelphia Water Department, Planning Office, etc.) that require accurate accounting of vacant parcels. These users need a tool that moves beyond static tax records to identify physical signatures of structural failure—roof collapses, bowing walls, or missing structural members. The output is a binary segmentation mask (Imminently Dangerous vs. Stable) at the parcel level, helping target users identify prime candidates for acquisition and stabilization before they cause irreparable injury to residents or adjacent structures.

---

## Technical Justification

Structural damage in high-resolution satellite imagery is defined by changes in geometry and texture rather than simple spectral signatures.

### Semantic Segmentation

To identify a building as "dangerous," a model must recognize localized failure points (e.g., a hole in a roof) within the context of the larger structure.

- **Task Selection:** We will utilize Semantic Segmentation powered by a pre-trained Building Damage Model (similar to post-disaster recovery models). By segmenting the specific areas of a roof or facade that show "damage," we can calculate a structural distress score per parcel.
- **Structural Distress Score:** By calculating the ratio of "damaged" pixels to the total building footprint, we can generate a normalized distress score.

### Failure Modes & Mitigation

- **New construction:** New roof installations can resemble holes or debris. We will mitigate this by filtering parcels with active eCLIPSE permits.
- **Parallax & Occlusion:** Philadelphia's tall "row home canyons" create shadows that hide facade leaning. To address this, we will integrate elevation data—either the latest Philadelphia Digital Elevation Model (DEM) or LiDAR-derived height models—to detect vertical deviations that RGB imagery might miss. We are evaluating the Philadelphia DEM as the preferred source for its local accuracy and recency.

---

## Methodological Precedent

- **The Clay Foundation Model (v1.5):** Rather than training a CNN from scratch, we use Clay, a Vision Transformer (ViT) pre-trained on global Earth Observation data. Clay's ability to understand urban textures allows for better generalization across Philadelphia's diverse neighborhoods, from Kensington's industrial shells to South Philadelphia's brick row houses.
  - Schroer et al. (2025) provides the foundations on the usage of Vision Transformer (ViT) models, such as the Clay model, instead of the traditional CNN processes. Through this approach, the computational overhead and amount of labeled data needed to extract spatial insights is significantly reduced.
  - Schroer, K., Adhikari, B., & Moise, I. (2025, May 29). *Revolutionizing earth observation with geospatial foundation models on AWS.* AWS Machine Learning Blog.

- **Building Footprint Decoupling:** Recent research suggests that decoupling localization (where is the building?) from classification (is it damaged?) improves accuracy in dense urban areas. We will use the VIDA Open Buildings dataset as our primary source of building footprint masks, with Google or Microsoft Open Buildings as potential supplements given their higher update frequency in the US. These static masks allow the model to focus purely on structural integrity.
  - Hatić et al. (2025) and Liu et al. (2021) both explore this approach.
  - Hatić, D., Polushko, V., Rauhut, M., & Hagen, H. (2025). *Post-Disaster Building Damage Assessment: Multi-Class Object Detection vs. Object Localization and Classification.* Remote Sensing, 17(24), 3957.
  - C. Liu, L. Ge and S. M. E. Sepasgozar, "Post-Disaster Classification of Building Damage Using Transfer Learning," 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, Brussels, Belgium, 2021, pp. 2194–2197.

- **Potential Model Improvements:** Although many existing building damage detection models are designed for post-disaster contexts, the classification challenges remain applicable to non-disaster settings. Model enhancements explored in related literature may inform our approach.
  - Ahmadi et al. (2023) discusses potential elements of a building damage detection model, including the use of adaptive kernel sizes and data augmentation to overcome the imbalance of buildings and background pixels.
  - Ahmadi, S. A., Mohammadzadeh, A., Yokoya, N., & Ghorbanian, A. (2024). *BD-SKUNet: Selective-Kernel UNets for Building Damage Assessment in High-Resolution Satellite Images.* Remote Sensing, 16(1), 182.

---

## Data Plan

- **Primary Imagery:** City of Philadelphia Aerial Imagery (most recent: 2025; past years available)
  - Resolution: 0.25 ft (7.62 cm)
  - Bands: 3-band (RGB)
  - Justification: Ultra-high resolution is essential for identifying precise details of the built environment.
- **Building Footprints:** VIDA Open Buildings dataset (primary), with Google or Microsoft Open Buildings as supplements for improved US coverage and recency. Parcel boundaries from the Philadelphia Water Department (PWD) parcel layers will supplement spatial joins to the administrative record.
- **Vector / Label Data:** L&I Vacant Lot violation data will serve as the primary source for training labels. Clean and seal data will be incorporated to improve ground-truth accuracy. Note that the VPI has recently resumed after a multi-year outage; we will incorporate it cautiously given ongoing data quality concerns.
- **Elevation Data:** Philadelphia Digital Elevation Model (DEM) for detecting vertical structural deviations. We are in contact with instructors regarding access to the latest extracted DEM.

---

## Modeling Approach

Our model will derive from a pre-built, global foundation model from Clay designed to efficiently distill and synthesize vast amounts of environmental data.

- **Structural Embedding (Clay):** For each building parcel, capture textural decay signals (rusting, debris, weathering).
- **Segmentation Head (UNet):** A UNet decoder will process these embeddings to produce a binary damage mask.
- **Temporal Change Detection:** By comparing 2024 and 2025 imagery, we identify the rate of structural decay. A building that shows a sudden darkening or texture change in the roof area over 12 months will be flagged for immediate inspection.

---

## Evaluation Strategy

Given the operational stakes of this model, we prioritize **Recall over Precision**. While there are an estimated 200–300 imminently dangerous buildings in Philadelphia at any given time, the cost of a false negative—a missed collapse that injures or kills a resident—is catastrophic, both in human terms and in terms of legal and policy liability. Even if the model over-identifies by 50%, the resulting volume of inspections (~450) remains operationally manageable for a small dedicated team. By contrast, missing a structural failure that leads to a wrongful death represents an irreversible policy failure. Accordingly, our primary optimization target is minimizing false negatives.

- **Primary Metric:** Mean Intersection over Union (mIoU) for the "Damaged" class.
- **The "ID" Goal:** A Recall of >0.90 for the "Imminently Dangerous" class. We prioritize capturing the vast majority of genuinely dangerous structures, accepting that some additional inspections will be required to clear false positives.
- **Secondary Metric:** Precision will be reported alongside Recall to characterize the false positive burden on inspection teams.
- **Geographic Stratification:** We will evaluate the model separately in North, South, and West Philadelphia to ensure that differences in architectural style (e.g., wood frame vs. masonry) do not bias results.

---

## Comparison of Considered Approaches

| Approach | Pro | Con |
|----------|-----|-----|
| NDVI Baseline | Finds "green" roofs (leaks leading to moss). | Misses structural failures without vegetation. |
| Random Forest | Fast; uses height features well. | Fails to capture the spatial "shape" of a collapse. |
| Clay + UNet | Captures decay textures; high recall with strong generalization. | High GPU demand; requires high-resolution imagery. |