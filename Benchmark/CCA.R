mtx_file=“./p22_rna_mtx/"
counts <- Read10X(mtx_file)
rna_obj <- CreateSeuratObject(counts = counts, project = "p22_rna")

mtx_file=“./p22_sprna_mtx/"
counts <- Read10X(mtx_file)
sprna_obj <- CreateSeuratObject(counts = counts, project = "p22_sprna")

rna<-rna_obj
sprna<-sprna_obj

rna <- NormalizeData(rna)
rna <- FindVariableFeatures(rna)

sprna <- NormalizeData(sprna)
sprna <- FindVariableFeatures(sprna)

features <- SelectIntegrationFeatures(object.list = list(rna, sprna))
rna <- ScaleData(rna)
sprna <- ScaleData(sprna)
rna_integrated <- RunCCA(rna, sprna, features = features)
cca_embeddings_rna <- Embeddings(rna_integrated, reduction = "cca")

write.table(cca_embeddings_rna, file = “./cca_p22.tsv")