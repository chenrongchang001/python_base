import array
import pysam
import argparse
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                    datefmt='%Y/%m/%d %H:%M:%S', filename='test.log', filemode='w')
args = None
def arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-b", "--base_quality", help='Minimum base quality', type=int, default=0)
    parser.add_argument("-m", "--map_quality", help='Minimum mapping quality', type=int, default=0)
    parser.add_argument("-n", "--normal_bam", help='Input normal bam file ', type=str)
    parser.add_argument("ref", help='Input reference fasta file')
    parser.add_argument("vcf", help='Input vcf format file')
    parser.add_argument("bam", help='Input bam file')
    parser.add_argument("out", help='Output file')
    args = parser.parse_args()
    return args


def args_test():
    logging.info(f'Vcf file is: {args.vcf}')
    logging.info(f'Bam file is: {args.bam}')
    if args.normal_bam:
        logging.info(f'Matched normal bam file is: {args.normal_bam}')
    logging.info(f'Base quality threshold is: {args.base_quality}')
    logging.info(f'Mapping quality threshold is: {args.map_quality}')
    logging.info(f'Reference fasta file is: {args.ref}')


class Feature:
    def __init__(self, record):
        self.contig, self.start, self.stop, self.ref, self.alt = record
        self.is_snv, self.is_insert, self.is_deletion = False, False, False
        if len(self.ref) == len(self.alt) == 1:
            self.is_snv = True
        elif len(self.ref) == 1 and len(self.alt) > 1:
            self.is_insert = True
        elif len(self.ref) == 1 and len(self.alt) > 1:
            self.is_deletion = True
        self.depth = 0
        self.alt_depth = 0
        self.alt_plus = 0
        self.alt_minus = 0
        self.alt_soft_clip = 0
        self.alt_position = array.array('f')
        self.alt_base_quality = array.array('I')
        self.alt_map_quality = array.array('I')
        self.alt_mismatch_base_position = array.array('I')
        self.alt_mismatch_base_reference_position = array.array('I')
        self.alt_mismatch_base_quality = array.array('I')
        self.alt_mismatch_base_distance_to_mutation = array.array('I')
        self.alt_mismatch_base_nearby = 0
        self.alt_insert_count = array.array('I')
        self.alt_delete_count = array.array('I')
        self.ref_depth = 0
        self.ref_plus = 0
        self.ref_minus = 0
        self.ref_soft_clip = 0
        self.ref_position = array.array('f')
        self.ref_base_quality = array.array('I')
        self.ref_map_quality = array.array('I')
        self.ref_mismatch_base_position = array.array('I')
        self.ref_mismatch_base_reference_position = array.array('I')
        self.ref_mismatch_base_quality = array.array('I')
        self.ref_mismatch_base_distance_to_mutation = array.array('I')
        self.ref_mismatch_base_nearby = 0
        self.ref_insert_count = array.array('I')
        self.ref_delete_count = array.array('I')
        self.third_base_depth = 0
        self.third_base_quality = array.array('I')
        self.deletion_depth = 0
        self.no_md_tag_reads_count = 0
        self.normal_depth = 0
        self.normal_alt_depth = 0

    def add_read(self, read, map_quality, base_quality, ref):
        if read.is_duplicate or read.is_secondary or read.is_supplementary or read.is_unmapped:
            return
        if read.mapping_quality < map_quality:
            return
        aligned_pairs = read.get_aligned_pairs()
        query_sequence = read.query_sequence
        query_qualities = read.query_qualities
        reference_start = read.reference_start
        query_alignment_start = read.query_alignment_start
        query_alignment_end = read.query_alignment_end
        cigartuples = read.cigartuples
        read_position = self.query_position(aligned_pairs)
        if read_position is None:   # is an deletion position in this reads
            self.deletion_depth += 1
            self.depth += 1
        else:
            base = query_sequence[read_position].upper()  # Base in this read of mutation position
            base_quality = query_qualities[read_position]
            if base_quality < base_quality:
                return
            self.depth += 1
            map_quality = read.mapping_quality
            insert_count, delete_count = self.indel_flanking(cigartuples, read_position)
            is_plus = not read.is_reverse
            is_minus = read.is_reverse
            soft_clip = 'S' in read.cigarstring
            if is_plus:
                read_position_percent = (read_position - query_alignment_start) / (query_alignment_end - query_alignment_start)
            else:
                read_position_percent = (query_alignment_end - read_position) / (query_alignment_end - query_alignment_start)
            if read.has_tag('MD'):
                md_tag = read.get_tag('MD')
                mismatch_query_point, mismatch_reference_point = self.mismatch_position(md_tag, reference_start, aligned_pairs)
            else:
                mismatch_query_point, mismatch_reference_point = self.mismatch_position_nomdtag(aligned_pairs, ref, query_sequence)
            mismatch_base_quality = [query_qualities[i] for i in mismatch_query_point]
            if base == self.alt.upper():
                self.alt_depth += 1
                self.alt_plus += int(is_plus)
                self.alt_minus += int(is_minus)
                self.alt_soft_clip += int(soft_clip)
                self.alt_position.append(read_position_percent)
                self.alt_base_quality.append(base_quality)
                self.alt_map_quality.append(map_quality)
                self.alt_mismatch_base_distance_to_mutation.extend([abs(i-read_position) for i in mismatch_query_point])
#                self.alt_mismatch_base_distance_to_mutation = [i for i in self.alt_mismatch_base_distance_to_mutation if i != 0]
                if 1 in [abs(i-read_position) for i in mismatch_query_point]:
                    self.alt_mismatch_base_nearby += 1
                alt_position_index = mismatch_query_point.index(read_position)
                del mismatch_query_point[alt_position_index], mismatch_base_quality[alt_position_index]
                del mismatch_reference_point[alt_position_index]
                self.alt_mismatch_base_reference_position.extend(mismatch_reference_point)
                self.alt_mismatch_base_position.extend(mismatch_query_point)
                self.alt_mismatch_base_quality.extend(mismatch_base_quality)
                self.alt_insert_count.append(insert_count)
                self.alt_delete_count.append(delete_count)
            elif base == self.ref.upper():
                self.ref_depth += 1
                self.ref_plus += int(is_plus)
                self.ref_minus += int(is_minus)
                self.ref_soft_clip += int(soft_clip)
                self.ref_position.append(min(read_position_percent, 1-read_position_percent))
                self.ref_base_quality.append(base_quality)
                self.ref_map_quality.append(map_quality)
                self.ref_mismatch_base_distance_to_mutation.extend([abs(i-read_position) for i in mismatch_query_point])
#                self.ref_mismatch_base_distance_to_mutation = [i for i in self.ref_mismatch_base_distance_to_mutation if i != 0]
                if 1 in [abs(i-read_position) for i in mismatch_query_point]:
                    self.ref_mismatch_base_nearby += 1
                self.ref_mismatch_base_reference_position.extend(mismatch_reference_point)
                self.ref_mismatch_base_position.extend(mismatch_query_point)
                self.ref_mismatch_base_quality.extend(mismatch_base_quality)
                self.ref_insert_count.append(insert_count)
                self.ref_delete_count.append(delete_count)
            else:
                self.third_base_depth += 1
                self.third_base_quality.append(base_quality)

    def add_read_from_normal(self, read, map_quality, base_quality):
        if read.is_duplicate or read.is_secondary or read.is_supplementary or read.is_unmapped:
            return
        if read.mapping_quality < map_quality:
            return
        aligned_pairs = read.get_aligned_pairs()
        read_position = self.query_position(aligned_pairs)
        query_sequence = read.query_sequence
        query_qualities = read.query_qualities
        if read_position is None:   # is an deletion position in this reads
            self.normal_depth += 1
        else:
            base = query_sequence[read_position].upper()  # Base in this read of mutation position
            base_quality = query_qualities[read_position]
            if base_quality < base_quality:
                return
            self.normal_depth += 1
            if base == self.alt.upper():
                self.normal_alt_depth += 1

    def mismatch_position_nomdtag(self, aligned_pairs, reference_fasta, query_sequence):
        ap = aligned_pairs.copy()
        ap1 = aligned_pairs.copy()
        for i, j in ap:
            if i is None or j is None:
                ap1.remove((i, j))
            else:
                break
        ap = ap1.copy()
        for i, j in ap[::-1]:
            if i is None or j is None:
                ap1.remove((i, j))
            else:
                break
        with pysam.FastaFile(reference_fasta) as fasta_file_obj:
            reference_sequence = fasta_file_obj.fetch(self.contig, ap1[0][-1], ap1[-1][-1]+1).upper()
        mismatch_reference_point = array.array('I')
        mismatch_query_point = array.array('I')
        for i, j in ap1:
            if i is None or j is None:
                pass
            else:
                point = j - ap1[0][-1]
                try:
                    if reference_sequence[point] != query_sequence[i].upper(): # Mismatch base position
                        mismatch_query_point.append(i)
                        mismatch_reference_point.append(j)
                except:
                    print(self.contig, ap1[0][-1], ap1[-1][-1]+1)
                    print(reference_sequence,point,query_sequence,i)
        return mismatch_query_point, mismatch_reference_point

    def mismatch_position(self, md_tag, reference_start, aligned_pairs):
        re_compile = re.compile(r'\^[a-zA-Z]+|[a-zA-Z]+')
        match = re_compile.split(md_tag)
        nomatch = re_compile.findall(md_tag)
        mismatch_reference_point = array.array('I')
        mismatch_query_point = array.array('I')
        reference_point = reference_start-1
        for i, j in zip(match, nomatch):
            if j.startswith('^'):
                reference_point = reference_point + int(i) + len(j) - 1
            else:
                reference_point = reference_point + int(i) + len(j)
                mismatch_reference_point.append(reference_point)
        for each in aligned_pairs:
            if each[1] in mismatch_reference_point:
                mismatch_query_point.append(each[0])
        return mismatch_query_point, mismatch_reference_point

    def indel_flanking(self, cigartuples, read_position, flank=5):
        insert_count = 0
        delete_count = 0
        point = array.array('I', [0])
        for each in cigartuples:
            if each[0] == 2:
                pass
            else:
                point.append(each[1]+point[-1])
            if abs(point[-1]-read_position) <= 5 or abs(point[-2]-read_position) <= flank:
                if each[0] == 2:
                    delete_count += 1
                elif each[0] == 1:
                    insert_count += 1
        return insert_count, delete_count

    def query_position(self, aligned_pairs):
        ap_pos = [i for i in aligned_pairs if i[1] == self.start]
        if ap_pos:
            return ap_pos[0][0]

    def statistic(self):
        self.alt_soft_clip = self.alt_soft_clip/self.alt_depth if self.alt_depth > 0 else 0
        self.alt_mismatch_base_nearby = self.alt_mismatch_base_nearby/self.alt_depth if self.alt_depth > 0 else 0
        self.alt_average_position = np.mean(self.alt_position) if len(self.alt_position) > 0 else 0
        self.alt_average_base_quality = np.mean(self.alt_base_quality) if len(self.alt_base_quality) > 0 else 0
        self.alt_average_map_quality = np.mean(self.alt_map_quality) if len(self.alt_map_quality) > 0 else 0
        self.alt_average_mismatch_base_count = len(self.alt_mismatch_base_position)/self.alt_depth if self.alt_depth > 0 else 0
        self.alt_average_mismatch_base_quality = np.mean(self.alt_mismatch_base_quality) if self.alt_average_mismatch_base_count != 0 else 0
        self.alt_mismatch_base_distance_to_mutation = [i for i in self.alt_mismatch_base_distance_to_mutation if i != 0]
        self.alt_average_mismatch_base_distance_to_mutation = np.mean(self.alt_mismatch_base_distance_to_mutation) if len(self.alt_mismatch_base_distance_to_mutation) > 0 else 0
        self.alt_mismatch_base_reference_position_variation = len(set(self.alt_mismatch_base_reference_position)) / len(self.alt_mismatch_base_reference_position) if len(self.alt_mismatch_base_reference_position) > 0 else 0
        self.alt_average_insert_count = np.mean(self.alt_insert_count) if len(self.alt_insert_count) > 0 else 0
        self.alt_average_delete_count = np.mean(self.alt_delete_count) if len(self.alt_delete_count) > 0 else 0

        self.ref_soft_clip = self.ref_soft_clip / self.ref_depth if self.ref_depth > 0 else 0
        self.ref_mismatch_base_nearby = self.ref_mismatch_base_nearby / self.ref_depth if self.ref_depth > 0 else 0
        self.ref_average_position = np.mean(self.ref_position) if len(self.ref_position) > 0 else 0
        self.ref_average_base_quality = np.mean(self.ref_base_quality) if len(self.ref_base_quality) > 0 else 0
        self.ref_average_map_quality = np.mean(self.ref_map_quality) if len(self.ref_map_quality) > 0 else 0
        self.ref_average_mismatch_base_count = len(self.ref_mismatch_base_position)/self.ref_depth if self.ref_depth > 0 else 0
        self.ref_average_mismatch_base_quality = np.mean(self.ref_mismatch_base_quality) if self.ref_average_mismatch_base_count != 0 else 0
        self.ref_mismatch_base_distance_to_mutation = [i for i in self.ref_mismatch_base_distance_to_mutation if i != 0]
        self.ref_average_mismatch_base_distance_to_mutation = np.mean(self.ref_mismatch_base_distance_to_mutation) if len(self.ref_mismatch_base_distance_to_mutation) > 0 else 0
        self.ref_mismatch_base_reference_position_variation = len(set(self.ref_mismatch_base_reference_position)) / len(self.ref_mismatch_base_reference_position) if len(self.ref_mismatch_base_reference_position) > 0 else 0
        self.ref_average_insert_count = np.mean(self.ref_insert_count) if len(self.ref_insert_count) > 0 else 0
        self.ref_average_delete_count = np.mean(self.ref_delete_count) if len(self.ref_delete_count) > 0 else 0

        self.third_base_average_quality = np.mean(self.third_base_quality) if len(self.third_base_quality) > 0 else 0
        self.third_base_vaf = self.third_base_depth /self.depth if self.depth > 0 else 0
        self.deletion_vaf = self.deletion_depth/self.depth if self.depth > 0 else 0

        if self.no_md_tag_reads_count > 0:
            logging.info(f'Position<{self.contig}:{self.start}> has {self.no_md_tag_reads_count} reads without MD tag.')
        self.in_normal = self.normal_alt_depth/self.normal_depth if self.normal_depth > 0 else 0

class VariantRecord:
    def __init__(self, *record):
        self.contig, self.start, self.stop, self.ref, self.alt = record
        self.feature = Feature(record)

    def __repr__(self):
        return(f'''VariantRecord({self.contig}, {self.start}, {self.stop}, {self.ref}, {self.alt})''')

    def get_feature_with_normal(self, t_reads_obj, n_reads_obj, map_quality, base_quality, ref):
        for read in t_reads_obj:
            self.feature.add_read(read, map_quality, base_quality, ref)
        for read in n_reads_obj:
            self.feature.add_read_from_normal(read, map_quality, base_quality)
        self.feature.statistic()

    def get_feature_without_normal(self, t_reads_obj, map_quality, base_quality, ref):
        for read in t_reads_obj:
            self.feature.add_read(read, map_quality, base_quality, ref)
        self.feature.statistic()

class Variant:
    '''Input is variant file which is vcf or other format'''
    def __init__(self, variant_file, file_format='vcf'):
        self.processed_variants_count = 0
        if file_format == 'vcf':
            self.variants = self.get_variant_from_vcf(variant_file)
        elif file_format == 'txt':
            self.variants = self.get_variant_from_txt(variant_file)
        self.all_variants_count = len(self)

    def __len__(self):
        return len(self.variants)

    def get_variant_from_vcf(self, variant_file):
        variants_list = []
        with pysam.VariantFile(variant_file) as vcf:
            variants = vcf.fetch()
            for variant in variants:
                variants_list.append(VariantRecord(variant.contig,
                                                   variant.start - 1,
                                                   variant.stop,
                                                   variant.ref.upper(),
                                                   variant.alt[0].upper()))
        return variants_list

    def get_variant_from_txt(self, variant_file):
        variants_list = []
        with open(variant_file) as txt:
            for variant in txt:
                if variant.startswith('chr'):
                    variant = variant.strip().split()
                    variants_list.append(VariantRecord(variant[0],
                                                       int(variant[1]) - 1,
                                                       int(variant[1]),
                                                       variant[2].upper(),
                                                       variant[3].upper()))
        return variants_list

    def get_features(self, bam_file, normal_bam_file, map_quality, base_quality, ref):
        if normal_bam_file:
            with pysam.AlignmentFile(bam_file, 'rb') as t_bam_obj:
                with pysam.AlignmentFile(normal_bam_file, 'rb') as n_bam_obj:
                    for variantrecord in self.variants:
                        if variantrecord.feature.is_snv:
                            t_reads_obj = t_bam_obj.fetch(variantrecord.contig, variantrecord.start, variantrecord.stop, until_eof=True)
                            n_reads_obj = n_bam_obj.fetch(variantrecord.contig, variantrecord.start, variantrecord.stop, until_eof=True)
                            variantrecord.get_feature_with_normal(t_reads_obj, n_reads_obj, map_quality, base_quality, ref)
                        self.processed_variants_count += 1
                        if self.processed_variants_count % 100 == 0:
                            logging.info(f'Processed {self.processed_variants_count} varaint record')
        else:
            with pysam.AlignmentFile(bam_file, 'rb') as t_bam_obj:
                for variantrecord in self.variants:
                    if variantrecord.feature.is_snv:
                        t_reads_obj = t_bam_obj.fetch(variantrecord.contig, variantrecord.start, variantrecord.stop, until_eof=True)
                        variantrecord.get_feature_without_normal(t_reads_obj, map_quality, base_quality, ref)
                    self.processed_variants_count += 1
                    if self.processed_variants_count % 100 == 0:
                        logging.info(f'Processed {self.processed_variants_count} varaint record')
        logging.info(f'Total processed <{self.processed_variants_count}/{self.all_variants_count}> varaint record')

    def show(self):
        for vr in self.variants:
            print(f'Chromosome:\t{vr.feature.contig}\n'
                  f'Start:\t{vr.feature.start}\n'
                  f'End:\t{vr.feature.stop}\n'
                  f'Ref:\t{vr.feature.ref}\n'
                  f'Alt:\t{vr.feature.alt}\n'
                  f'Depth:\t{vr.feature.depth}\n'
                  f'Alt_Depth:\t{vr.feature.alt_depth}\n'
                  f'Ref_Depth:\t{vr.feature.ref_depth}\n'
                  f'Third_Base_Depth:\t{vr.feature.third_base_depth}\n'
                  f'Delete_Depth:\t{vr.feature.deletion_depth}\n'
                  f'Alt_Plus:\t{vr.feature.alt_plus}\n'
                  f'Alt_Minus:\t{vr.feature.alt_minus}\n'
                  f'Alt_Position:\t{vr.feature.alt_average_position}\n'
                  f'Alt_Base_Quality:\t{vr.feature.alt_average_base_quality}\n'
                  f'Alt_Map_Quality:\t{vr.feature.alt_average_map_quality}\n'
                  f'Alt_Average_Mismatch_Count:\t{vr.feature.alt_average_mismatch_base_count}\n'
                  f'Alt_Average_Mismatch_Base_Quality:\t{vr.feature.alt_average_mismatch_base_quality}\n'
                  f'Alt_Insert_Count:\t{vr.feature.alt_average_insert_count}\n'
                  f'Alt_Delete_Count:\t{vr.feature.alt_average_delete_count}\n'
                  f'Ref_Plus:\t{vr.feature.ref_plus}\n'
                  f'Ref_Minus:\t{vr.feature.ref_minus}\n'
                  f'Ref_Position:\t{vr.feature.ref_average_position}\n'
                  f'Ref_Base_Quality:\t{vr.feature.ref_average_base_quality}\n'
                  f'Ref_Map_Quality:\t{vr.feature.ref_average_map_quality}\n'
                  f'Ref_Average_Mismatch_Count:\t{vr.feature.ref_average_mismatch_base_count}\n'
                  f'Ref_Average_Mismatch_Base_Quality:\t{vr.feature.ref_average_mismatch_base_quality}\n'
                  f'Ref_Insert_Count:\t{vr.feature.ref_average_insert_count}\n'
                  f'Ref_Delete_Count:\t{vr.feature.ref_average_delete_count}\n')

    def to_csv(self, csv_file):
        csv_dict = defaultdict(list)
        for vr in self.variants:
            if not vr.feature.is_snv:
                continue
            csv_dict['Chromosome'].append(vr.feature.contig)
            csv_dict['Start'].append(vr.feature.start)
            csv_dict['End'].append(vr.feature.stop)
            csv_dict['Ref'].append(vr.feature.ref)
            csv_dict['Alt'].append(vr.feature.alt)
            csv_dict['Depth'].append(vr.feature.depth)
            csv_dict['Alt_Depth'].append(vr.feature.alt_depth)
            csv_dict['Ref_Depth'].append(vr.feature.ref_depth)
            csv_dict['Third_Base_Vaf'].append(vr.feature.third_base_vaf)
            csv_dict['Third_Base_Average_Quality'].append(vr.feature.third_base_average_quality)
            csv_dict['Delete_Vaf'].append(vr.feature.deletion_vaf)
            csv_dict['Alt_Soft_Clip'].append(vr.feature.alt_soft_clip)
            csv_dict['Alt_Plus'].append(vr.feature.alt_plus)
            csv_dict['Alt_Minus'].append(vr.feature.alt_minus)
            csv_dict['Alt_Position'].append(vr.feature.alt_average_position)
            csv_dict['Alt_Base_Quality'].append(vr.feature.alt_average_base_quality)
            csv_dict['Alt_Map_Quality'].append(vr.feature.alt_average_map_quality)
            csv_dict['Alt_Average_Mismatch_Count'].append(vr.feature.alt_average_mismatch_base_count)
            csv_dict['Alt_Average_Mismatch_Base_Quality'].append(vr.feature.alt_average_mismatch_base_quality)
            csv_dict['Alt_Average_Mismatch_Base_Distance_To_Mut'].append(vr.feature.alt_average_mismatch_base_distance_to_mutation)
            csv_dict['Alt_Mismatch_Base_Reference_Position_Variation'].append(vr.feature.alt_mismatch_base_reference_position_variation)
            csv_dict['Alt_Mismatch_Base_Nearby'].append(vr.feature.alt_mismatch_base_nearby)
            csv_dict['Alt_Insert_Count'].append(vr.feature.alt_average_insert_count)
            csv_dict['Alt_Delete_Count'].append(vr.feature.alt_average_delete_count)
            csv_dict['Ref_Soft_Clip'].append(vr.feature.ref_soft_clip)
            csv_dict['Ref_Plus'].append(vr.feature.ref_plus)
            csv_dict['Ref_Minus'].append(vr.feature.ref_minus)
            csv_dict['Ref_Position'].append(vr.feature.ref_average_position)
            csv_dict['Ref_Base_Quality'].append(vr.feature.ref_average_base_quality)
            csv_dict['Ref_Map_Quality'].append(vr.feature.ref_average_map_quality)
            csv_dict['Ref_Average_Mismatch_Count'].append(vr.feature.ref_average_mismatch_base_count)
            csv_dict['Ref_Average_Mismatch_Base_Quality'].append(vr.feature.ref_average_mismatch_base_quality)
            csv_dict['Ref_Average_Mismatch_Base_Distance_To_Mut'].append(vr.feature.ref_average_mismatch_base_distance_to_mutation)
            csv_dict['Ref_Mismatch_Base_Reference_Position_Variation'].append(vr.feature.ref_mismatch_base_reference_position_variation)
            csv_dict['Ref_Mismatch_Base_Nearby'].append(vr.feature.ref_mismatch_base_nearby)
            csv_dict['Ref_Insert_Count'].append(vr.feature.ref_average_insert_count)
            csv_dict['Ref_Delete_Count'].append(vr.feature.ref_average_delete_count)
            csv_dict['Depth_in_normal'].append(vr.feature.normal_depth)
            csv_dict['Vaf_in_normal'].append(vr.feature.in_normal)
        csv_df = pd.DataFrame(csv_dict)
        csv_df.to_csv(csv_file, sep=',', index=False)
        return csv_df

if __name__ == '__main__':
    args = arguments()
    args_test()
    variants = Variant(args.vcf, file_format='txt')
    variants.get_features(args.bam, args.normal_bam, args.map_quality, args.base_quality, args.ref)
#    variants.show()
    variants.to_csv(args.out)
