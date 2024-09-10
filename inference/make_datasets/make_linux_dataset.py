import os
import json
import re

def remove_links(raw_crash_report) :
    line_list = []
    lines = raw_crash_report.split("\n")
    for line in lines :
        line = remove_href_links(line)
        line_list.append(line)
    new_crash_report = "\n".join(line_list)
    return new_crash_report

def remove_href_links(line) :
    pattern = re.compile(r'(<a href=.*>)(.*)(<\/a>)')
    # Search for the pattern in the text
    match = pattern.search(line)
    if match :
        left_brace = match.groups()[0]
        inside_content = match.groups()[1]
        right_brace = match.groups()[2]
        complete = left_brace+inside_content+right_brace
        line = line.replace(complete,inside_content)
    return line

def save_list(complete_data,path) :
    with open(os.path.join(path,"kernel_bench_data.json"),"w") as f :
        json.dump(complete_data,f,indent=4)

def get_crash_report_from_google_cloud_dump(google_cloud_data_path,
                                        job_id) :
    path_to_crash_report = os.path.join(google_cloud_data_path,job_id,"0_kvmmanager_report0")

    if not os.path.exists(path_to_crash_report) :
        path_to_crash_report = os.path.join(google_cloud_data_path,job_id,"1_kvmmanager_report0")

    with open(path_to_crash_report,"r") as f :
        crash_report = f.read()
    crash_report = crash_report.replace("root/work_dir/linux/","")
    return crash_report

def main() :
    ############################################################################
    golden_subset_path = os.getenv("GOLDEN_SUBSET_PATH")
    benchmark_path = os.getenv("KBENCH_PATH")
    linux_path = os.getenv("LINUX_PATH")
    google_cloud_data_path = os.path.join(os.getenv("BASE_PATH"),"google_cloud_data")

    assert(os.path.exists(golden_subset_path))
    assert(os.path.exists(benchmark_path))
    assert(os.path.exists(linux_path))
    assert(os.path.exists(google_cloud_data_path))
    ############################################################################

    golden_subset_data = json.load(open(golden_subset_path,"r"))
    final_ans = []
    for ele in golden_subset_data["bugs"] :
        bug_id = ele["id"]

        original_crash_report = None
        # original_job_id = ele["original_job_id"]
        # original_crash_report = get_crash_report_from_google_cloud_dump(google_cloud_data_path,original_job_id)

        parent_job_id = ele["parent_job_id"]
        parent_crash_report = get_crash_report_from_google_cloud_dump(google_cloud_data_path,parent_job_id)

        with open(os.path.join(benchmark_path,bug_id+".json")) as f :
            bug_data = json.load(f)

        # if bug_data.get("raw_crash_report",-1) == -1 :
        #     print("Bug : {}".format(bug_id))
        #     assert(False)
        
        # raw_crash_report = remove_links(bug_data["raw_crash_report"])
        kernel_source_commit = bug_data["crashes"][0]["kernel-source-commit"]
        parent_of_fix_commit = bug_data["parent_of_fix_commit"]
        kernel_patch = bug_data["patch"]
        oracle_files = bug_data["patch_modified_files"]
        temp_info = {
            "instance_id" : bug_id,
            "patch" : kernel_patch,
            "repo": linux_path,
            "base_commit" : kernel_source_commit,
            "parent_commit" : parent_of_fix_commit,
            "hints_text" : "",
            "created_at" : "",
            "test_patch" : "",
            "problem_statement" : "",
            "original_commit_problem_statement" : original_crash_report,
            "parent_commit_problem_statement" : parent_crash_report,
            "version" : "",
            "environment_setup_commit": "",
            "FAIL_TO_PASS" : "",
            "PASS_TO_PASS": "",
            "oracle_files" : oracle_files
        }
        final_ans.append(temp_info)

    save_list(complete_data=final_ans,path=os.path.join(os.getenv("KBENCH_EXPR_PATH"),"dataset"))

    return final_ans

if __name__ == '__main__' :
    main()