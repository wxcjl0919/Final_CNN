library(GenomicRanges)
library(DNAshapeR)
library(BSgenome.Hsapiens.UCSC.hg19)
library(Biostrings)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(grid)
setwd('/Users/xinzeng/Desktop/research/reproduce')

				#####################################################################
				##																   ##
				##																   ##
				##							FUNCTION REGION						   ##
				##																   ##
				##																   ##
				#####################################################################

merge_file_path <- function(file_dir,tss_pair_names){

	file_path <- paste(file_dir,tss_pair_names,sep = "")
	minus_file <- paste(file_path,"minus.bed",sep = '_')
	plus_file <- paste(file_path,"plus.bed",sep = '_')

	file_list <- list(x1 = minus_file, x2 = plus_file)

	return(file_list)
}

bed_to_grange <- function(bed_file_path){

	bed_file <- read.table(bed_file_path, header = FALSE)

	colnames(bed_file)[1:3] <- c('seqnames','start','end')
	
	grange_file <- GRanges(seqnames = bed_file$seqnames, 
		ranges = IRanges(start = bed_file$start, end = bed_file$end))
return(grange_file)
}

merge_grange <- function(minus_gr,plus_gr){

	merge_gr <- GRanges(seqnames = seqnames(minus_gr),ranges = IRanges(start = start(minus_gr), end = end(plus_gr)))
return(merge_gr)
}


find_overlap <- function(gr_file,annotate_gr_file){

	overlap_list <- findOverlaps(annotate_gr_file,gr_file)
	overlap <- unique(gr_file[subjectHits(overlap_list)])

	return(overlap)
}


sub_group <- function(gr_file,annotated_gr,sub_type){

	match_index <- c()

	if (sub_type == 'start') {

		gr_start <- start(gr_file)
		annotated_start	 <- start(annotated_gr)

		for (x in 1:length(annotated_start)){
			match_index <- c(match_index,match(annotated_start[x],gr_start))
		}
	}

	else if(sub_type == 'end'){

		gr_end <- end(gr_file)
		annotated_end	 <- end(annotated_gr)

		for (x in 1:length(annotated_end)){
			match_index <- c(match_index,match(annotated_end[x],gr_end))
		}
	}

return(gr_file[match_index])	

}	

center_gr <- function(minus_gr,plus_gr){

	df <- as(minus_gr,'data.frame')

	df$start <- (start(minus_gr)+end(minus_gr))/2
	df$end <- (start(plus_gr)+end(plus_gr))/2

	df <- df[,-4]

return(df)
}


get_fasta <- function(df,length,saved_file_name){


	df$center <- (df$start + df$end)/2
	df$start <- (df$center - length/2)
	df$end <-  (df$center + length/2)
	df <- df[,-4]

	df_gr <- GRanges(df)

	getFasta(df_gr,BSgenome.Hsapiens.UCSC.hg19,width = length,filename = saved_file_name)

	print('successful saved!')
}


tss_distribution <- function(minus_gr,plus_gr){

	minus_tss <- end(minus_gr) - start(minus_gr)
	plus_tss <- end(plus_gr) - start(plus_gr)	
	tss_distance <- (start(plus_gr)+end(plus_gr))/2 - (start(minus_gr)+end(minus_gr))/2

	df <- data.frame(minus_tss,tss_distance,plus_tss)
	return(df)
}
	


plot_histogram <- function(df,column_name,x_lab){

	mean <- mean(column_name, na.rm=T)


	p1 <- ggplot(df,aes(x=column_name)) + geom_histogram(binwidth=10, color = 'black',fill = 'white')
	p1 <- p1 + geom_vline(aes(xintercept= mean),color="red", linetype="dashed", size=1)

	p1 <- p1 + xlim(0,400) + ylim(0,220)
	p1 <- p1 + ylab('Frequency')+ xlab(x_lab)

	p1 <- p1 + annotate('text', x = mean+80,y=200,label = paste('mean=',sprintf("%0.1f",mean),'bp',sep = ""),parse =FALSE)

	return(p1)
}




				#####################################################################
				##																   ##
				##																   ##
				##						 COMMAND LINE REGION					   ##
				##																   ##
				##																   ##
				#####################################################################


#set the data file
data_dir <- '41588_2014_BFng3142_MOESM78_ESM/'

# change the name here to get different input files
tss_pair_name <- 'tss_UU_k562'
annotation <- 'K562HMM_strong_enhancer.bed'


#load the bed file and transfer it to grange format
file_list <- merge_file_path(data_dir,tss_pair_name)
minus_gr <- bed_to_grange(file_list$x1)
plus_gr <- bed_to_grange(file_list$x2)
annotation_gr <- bed_to_grange(annotation)

remove(file_list)


# merge the minus TSS and plus TSS
merge_gr <- merge_grange(minus_gr,plus_gr)

annotated_tss <- find_overlap(merge_gr,annotation_gr)

annotated_tss_minus <- sub_group(minus_gr,annotated_tss,'start')
annotated_tss_plus <- sub_group(plus_gr,annotated_tss,'end')


annotated_tss_df <- tss_distribution(annotated_tss_minus,annotated_tss_plus)

#save annotated_tss as tsv format
center_gr_df <- center_gr(annotated_tss_minus,annotated_tss_plus)
write.table(center_gr_df, file='/Users/xinzeng/Desktop/research/result/5_27_2020/K562_UU', quote=FALSE, sep='\t')

#save fasta sequence of annotated_tss
get_fasta(center_gr_df,500,'/Users/xinzeng/Desktop/research/result/5_27_2020/K562_UU.fa')

# plot
annotated_tss_df <- tss_distribution(annotated_tss_minus,annotated_tss_plus)
p1 <- plot_histogram(annotated_tss_df,annotated_tss_df$minus_tss,'minus_tss')
p2 <- plot_histogram(annotated_tss_df,annotated_tss_df$tss_distance,'tss_distance')
p3  <- plot_histogram(annotated_tss_df,annotated_tss_df$plus_tss,'plus_tss')
grid.arrange(p1,p2,p3,ncol = 3)

