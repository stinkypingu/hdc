# hdc
hyperdimensional computing

Currently using Lit-PCBA dataset for the GBA protein(?) and attempts to classify molecules whether they actively/inactively bind to GBA. 

Methodology so far, assigns hypervectors to characters in the dataset strings. Groups by bigrams/trigrams within a molecule by binding vectors, then bundles together vectors for a hdv representation of the whole molecule. Bundles together 80% of the data to make a hypervector profile of active molecules and another for inactive molecules, then compares new molecules' hypervector representation to these two to make a classification.
