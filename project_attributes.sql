--https://donorschoose.looker.com/explore/main/project?qid=yudlo49lfa5Yyglrunz6my

SELECT
    project.project_id  AS "project_id",
        (DATE(CONVERT_TIMEZONE('UTC', 'America/New_York', project_workflow_facts.last_funded_at ))) AS "funded_date",
    project.number_of_materials_vendors  AS "materials_vendors_count",
    project.resource_subtotal  AS "material_cost",
        (CASE WHEN proposalmatching.proposalid IS NOT NULL THEN 'Yes' ELSE 'No' END) AS "has_match_on_posting",
        (CASE WHEN coalesce(project.is_fy25_equity_focus_school_at_time_of_posting,
                  project.is_fy25_equity_focus_school_at_time_of_submission,
                  project.is_fy25_equity_focus_school_at_time_of_creation) THEN 'Yes' ELSE 'No' END) AS "fy25_historical_efs_status",
        (DATE(CONVERT_TIMEZONE('UTC', 'America/New_York', project.expired_at ))) AS "expiration_date",
        (CASE WHEN project.is_a_plus OR (exciting_project.proposalid IS NOT NULL)  THEN 'Yes' ELSE 'No' END) AS "is_favorite_or_exciting",
        (CASE WHEN project_search_tag_facts.is_professional_development  THEN 'Yes' ELSE 'No' END) AS "is_professional_development",
        (CASE WHEN project.is_student_led  THEN 'Yes' ELSE 'No' END) AS "is_student_led",
        (CASE WHEN project_workflow_facts.teachers_nth_posted_project = 1  THEN 'Yes' ELSE 'No' END) AS "is_teachers_first_posted_project",
    project_workflow_facts.teachers_nth_posted_project  AS "teachers_nth_posted_project",
        (CASE WHEN project.is_one_time_sponsor_donor_a_government_entity  THEN 'Yes' ELSE 'No' END) AS "project_received_government_grant",
    project.percentage_free_lunch_at_time_of_posting  AS "percentage_free_lunch_at_time_of_posting",
    CASE
WHEN project.metro_type_at_time_of_posting = 'R'  THEN 'Rural'
WHEN project.metro_type_at_time_of_posting = 'T'  THEN 'Town'
WHEN project.metro_type_at_time_of_posting = 'S'  THEN 'Suburban'
WHEN project.metro_type_at_time_of_posting = 'U'  THEN 'Urban'
ELSE 'Unclassified'
END AS "metro_type_at_time_of_posting",
    SUBSTRING(school_address.zip, 1, 5)  AS "school_zip",
    project.school_id_at_time_of_posting AS "school_id_at_time_of_posting",
    school.enrollment AS "school_enrollment",
        (CASE WHEN school_mdr.is_historically_underrepresented_race  THEN 'Yes' ELSE 'No' END) AS "school_is_historically_underrepresented_race",
        (CASE WHEN school_mdr.is_low_income  THEN 'Yes' ELSE 'No' END) AS "school_is_low_income",
        (CASE WHEN school_mdr.is_racially_predominant  THEN 'Yes' ELSE 'No' END) AS "school_is_racially_predominant",
        (CASE WHEN school_mdr.is_underserved_rural THEN 'Yes' ELSE 'No' END) AS "school_is_underserved_rural",
    school_mdr.school_year_open  AS "school_year_open",
    coalesce(school_mdr.percent_black_imputed,0)  AS "school_percent_black_imputed",
    coalesce(school_mdr.percent_latinx_imputed,0)  AS "school_percent_latinx_imputed",
    coalesce(school_mdr.percent_asian_imputed,0)  AS "school_percent_asian_imputed",
    coalesce(school_mdr.percent_white_imputed,0)  AS "school_percent_white_imputed",
        (DATE(CONVERT_TIMEZONE('UTC', 'America/New_York', sf_district_account.discouraging_dc_org_1_c))) AS "sf_district_account_date_discouraging_dc_date_1",
        (DATE(teacher.created)) AS "teacher_created_date",
    CAST(EXTRACT(day from CAST(CONVERT_TIMEZONE('UTC', 'America/New_York', case when teacher_facts.first_project_posted_at < teacher_facts.created_at then null else teacher_facts.first_project_posted_at end ) AS TIMESTAMP) - CAST(CONVERT_TIMEZONE('UTC', 'America/New_York', teacher_facts.created_at ) AS TIMESTAMP)) AS BIGINT) AS "teacher_days_to_activation",
        (DATE(CONVERT_TIMEZONE('UTC', 'America/New_York', teacher_facts.first_project_posted_at ))) AS "teacher_first_project_posted_date",
        (CASE WHEN teacher.photopublished  THEN 'Yes' ELSE 'No' END) AS "teacher_photo_published",
        (CASE WHEN teacher_facts.number_of_teacher_profile_photos > 0  THEN 'Yes' ELSE 'No' END) AS "teacher_has_profile_photo",
    CASE
WHEN teacher_current_demographics.is_teacher_of_color is true  THEN 'Yes'
WHEN teacher_current_demographics.is_teacher_of_color is false  THEN 'No'
ELSE 'unknown'
END AS "teacher_is_teacher_of_color",
    teacher_current_demographics.gender  AS "teacher_gender",
    teacher_current_demographics.start_teaching_year  AS "teacher_start_teaching_year",
        (DATE(CONVERT_TIMEZONE('UTC', 'America/New_York', project_workflow_facts.first_posted_at ))) AS "posted_date",
    CASE
WHEN project_labels.grade = 'Grades PreK-2'  THEN 'Grades PreK-2'
WHEN project_labels.grade = 'Grades 3-5'  THEN 'Grades 3-5'
WHEN project_labels.grade = 'Grades 6-8'  THEN 'Grades 6-8'
WHEN project_labels.grade = 'Grades 9-12'  THEN 'Grades 9-12'
WHEN project_labels.grade is null  THEN ''
ELSE 'unknown'
END AS "grade_band",
    case when project_labels.category not in ('Other','Technology','Supplies') then project_labels.category end AS "project_category",
    project.total_cost AS "total_cost",
    teacher.personid  AS "teacher_id"
FROM dbt_target.fct_projects  AS project
LEFT JOIN ${project_labels.SQL_TABLE_NAME} AS project_labels ON project.project_id = project_labels.projectid
LEFT JOIN dbt_target.fct_project_workflows  AS project_workflow_facts ON project.project_id = project_workflow_facts.project_id
LEFT JOIN ${exciting_project.SQL_TABLE_NAME} AS exciting_project ON project.project_id = exciting_project.proposalid
LEFT JOIN dbt_target.fct_project_search_tags_aggregated  AS project_search_tag_facts ON project.project_id = project_search_tag_facts.project_id
LEFT JOIN ${proposalmatching_audit.SQL_TABLE_NAME} AS eligible_matching_project ON eligible_matching_project.proposalid = project.project_id
LEFT JOIN matching  AS project_match_eligible ON project_match_eligible.matchingid = eligible_matching_project.matchingid
LEFT JOIN dcteacher  AS teacher_ ON teacher_.teacherid = project.teacher_id
LEFT JOIN ${users.SQL_TABLE_NAME} AS teacher ON teacher_.teacherid = teacher.personid
LEFT JOIN dbt_target.fct_teachers  AS teacher_facts ON teacher_.teacherid = teacher_facts.teacher_id
LEFT JOIN school  AS school ON school.schoolid = project.school_id
LEFT JOIN dbt_target.school_demographics  AS school_mdr ON school_mdr.school_id = school.schoolid
LEFT JOIN address  AS school_address ON school.addressid = school_address.addressid
LEFT JOIN dbt_target.fct_school_organizations  AS district_school_education_organization ON district_school_education_organization.school_id = school.schoolid
LEFT JOIN ${sf_account_audit.SQL_TABLE_NAME} AS sf_district_account ON sf_district_account.district_id_c = district_school_education_organization.education_organization_id
LEFT JOIN dbt_target.fct_joined_resources  AS joined_resources ON project.project_id = joined_resources.project_id
LEFT JOIN dbt_target.fct_ariba_resources  AS ariba_resource ON ariba_resource.ariba_resource_id = joined_resources.ariba_resource_id
LEFT JOIN dbt_target.teacher_current_demographics  AS teacher_current_demographics ON teacher_current_demographics.teacher_id = teacher_.teacherid
LEFT JOIN proposalmatching on project.project_id = proposalmatching.proposalid AND date(proposalmatching.eligibilitytimestamp) = date(project.first_posted_at)
WHERE (NOT (project.is_essentials_list ) OR (project.is_essentials_list ) IS NULL) AND (NOT ((UPPER(school_address.state)) = 'PR' ) OR ((UPPER(school_address.state)) = 'PR' ) IS NULL) AND ((project.project_id ) > 0 OR (project.project_id ) IS NULL) 
  AND project_workflow_facts.first_posted_at >= CONVERT_TIMEZONE('America/New_York', 'UTC', TIMESTAMP '2024-08-22')
GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41