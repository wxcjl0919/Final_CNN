# The process detail of my cnn model

1. Download the TSS pair data from the Supplementary Data Set of published paper.
	
	Core, L. J., Martins, A. L., Danko, C. G., Waters, C. T., Siepel, A., & Lis, J. T. (2014). Analysis of nascent RNA identifies a unified architecture of initiation regions at mammalian promoters and enhancers. Nature Genetics, 46(12), 1311–1320. https://doi.org/10.1038/ng.3142

	File name  : 41588_2014_BFng3142_MOESM78_ESM

2. Download active promoters and strong enhancers annotation bed files from ChromHMM for K562 cell and GM12878 cell.
	
	Websites: http://genome.ucsc.edu/cgi-bin/hgFileUi?db=hg19&g=wgEncodeBroadHmm
	File name : wgEncodeBroadHmmGm12878HMM.bed, wgEncodeBroadHmmK562HMM.bed

3. Extract active promoters and strong enhancers information from ChromHMM annotation file.
	
	grep '255,0,0' wgEncodeBroadHmmGm12878HMM.bed > GM12878_ActivePromoters.bed
	grep '250,202,0' wgEncodeBroadHmmGm12878HMM.bed > GM12878_StrongEnhancers.bed
	grep '250,202,0' wgEncodeBroadHmmK562HMM.bed > K562_StrongEnhancers.bed
	grep '255,0,0' wgEncodeBroadHmmK562HMM.bed > K562_ActivePromoters.bed

4