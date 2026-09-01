# Experimental Results

## Experimental Setup

The project uses approximately 10,000 procedurally generated
basketball-trophy-inspired 3D shapes.

Each object is represented as a 32×32×32 binary occupancy grid.

The dataset is divided deterministically into:

- 80% training
- 10% validation
- 10% testing

Both the autoencoder and variational autoencoder use a
32-dimensional latent representation.

## Reconstruction

The deterministic autoencoder achieved:

- BCE: 0.000980
- IoU: 0.9946
- Dice: 0.9973

The variational autoencoder achieved:

- BCE: 0.002329
- IoU: 0.9870
- Dice: 0.9934

Both models reconstruct unseen trophy geometry accurately, although the
deterministic autoencoder performs better on all three reconstruction
metrics.

## Representation

Linear regression probes were trained to recover known procedural
parameters from the learned autoencoder latent representation.

The strongest results were:

- lower base radius: R² = 0.9590
- ball radius: R² = 0.9474
- body bottom radius: R² = 0.9397
- body top radius: R² = 0.9299
- support sweep: R² = 0.9025
- body height: R² = 0.8631

These results show that several major geometric properties are encoded
in a linearly accessible form.

Other properties were substantially weaker, demonstrating that latent
interpretability is not uniform across all procedural variables.

## Controllable Editing

Linear probe coefficients were used as candidate semantic directions in
latent space.

Moving encoded trophies along these directions produced controllable
changes for several strongly encoded properties.

The disentanglement experiment additionally measured how each edit
affected the other known geometric properties.

This distinction is important: strong property prediction does not by
itself imply independent geometric control.

## Shape Optimization

A differentiable latent-space objective was used to increase predicted
lower-base radius while decreasing predicted ball radius.

The optimization changed:

- predicted lower-base radius from 1.2643 to 1.2923
- predicted ball radius from 0.9230 to 0.7578

The decoded geometry remained coherent throughout the optimization
trajectory.

This demonstrates that the learned representation can support
goal-directed design modification in addition to reconstruction and
interpolation.

## VAE Generation

Standard-normal sampling produced:

- mean largest-component fraction: 0.4662
- fully connected samples: 9%
- more than 99% of voxels in the largest component: 12%
- more than 95% of voxels in the largest component: 15%

Reducing sampling standard deviation substantially improved connectivity.

At standard deviation 0.7:

- mean largest-component fraction: 0.9032
- fully connected samples: 66%
- more than 99% of voxels in the largest component: 73%
- more than 95% of voxels in the largest component: 75%

At standard deviation 0.5:

- mean largest-component fraction: 0.9695
- fully connected samples: 86%
- more than 99% of voxels in the largest component: 93%
- more than 95% of voxels in the largest component: 94%

Reduced-variance sampling should not be interpreted as standard VAE
prior sampling. Instead, these results indicate that the learned
aggregate posterior does not perfectly match the standard-normal prior.
Sampling closer to the center of latent space substantially improves
geometric coherence.

## Research Questions

### Can the models reconstruct unseen trophy geometry?

Yes. Both models achieve high reconstruction accuracy, with the
deterministic autoencoder performing best.

### Can the VAE generate novel trophy-like geometry?

Yes, but generation quality depends strongly on the sampled latent
region. Standard-normal sampling produces a significant fraction of
disconnected geometry.

### Does latent space encode known geometric properties?

Yes for several major properties. Six procedural variables achieve
strong linear predictability, while others are considerably weaker.

### Can latent directions control geometry?

Yes. Strong probe directions produce interpretable geometric changes,
although the representation is not perfectly disentangled.

### Can the learned representation support optimization?

Yes. Gradient-based optimization can move an encoded trophy toward a
desired geometric objective while maintaining coherent decoded geometry.

## Limitations

The main limitations are:

1. 32³ occupancy grids provide limited geometric resolution.
2. Some procedural variables are poorly represented by linear probes.
3. Semantic latent directions are not perfectly disentangled.
4. Standard VAE prior sampling does not consistently generate connected
   geometry.
5. The procedural trophy distribution is controlled and substantially
   simpler than general real-world CAD geometry.

## Conclusion

The experiments demonstrate that a learned 3D latent representation can
capture several known geometric properties in an accessible form and
can support interpolation, controllable editing, and differentiable
shape optimization.

The deterministic autoencoder provides the strongest reconstruction and
most useful representation for controllable editing, while the VAE adds
a generative sampling mechanism at the cost of weaker reconstruction and
less reliable standard-prior generation.