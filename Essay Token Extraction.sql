WITH base_projects AS (
    SELECT
        project.project_id AS project_id,
        project_essays.essay AS essay
    FROM dbt_target.fct_projects AS project
    LEFT JOIN dbt_target.fct_project_text AS project_essays
        ON project.project_id = project_essays.project_id
    LEFT JOIN dbt_target.fct_project_workflows AS project_workflow_facts
        ON project.project_id = project_workflow_facts.project_id
    WHERE
        (NOT project.is_essentials_list OR project.is_essentials_list IS NULL)
        AND project_workflow_facts.last_content_or_resource_approved_at >= CONVERT_TIMEZONE('America/New_York', 'UTC', TIMESTAMP '2026-01-30')
        AND project.project_id > 0
        -- retains only x% of projects
        AND project.project_id % 50 = 2
),
numbers AS (
    SELECT ROW_NUMBER() OVER ()::INT AS n
    FROM dbt_target.fct_projects
    LIMIT 500
)
SELECT
    project_id,
    LISTAGG(token, ',') WITHIN GROUP (ORDER BY term_count DESC) AS tokens
FROM (
    SELECT
        bp.project_id,
        LOWER(
            SPLIT_PART(
                REGEXP_REPLACE(bp.essay, '[^a-zA-Z]+', ' '),
                ' '::VARCHAR,
                n.n
            )
        ) AS token,
        COUNT(*) AS term_count
    FROM base_projects bp
    JOIN numbers n
        ON n.n <= REGEXP_COUNT(
            REGEXP_REPLACE(bp.essay, '[^a-zA-Z]+', ' '),
            ' '
        ) + 1
    WHERE bp.essay IS NOT NULL
    GROUP BY
        bp.project_id,
        token
) t
WHERE
    -- only terms 3+ characters or 'ai'
    (LENGTH(token) >= 3 OR token = 'ai')

    -- static stopword removal
    AND token NOT IN 
        -- articles / conjunctions / prepositions
        ('the','and','for','with','are','was','were','been','being',
        'that','this','these','those','from','into','about','over',
        'under','after','before','between','while','than','then',
        'because','through','during','without','within',

        -- pronouns / possessives
        'they','them','their','theirs','we','our','ours','you','your',
        'yours','i','me','my','mine','he','him','his','she','her','hers',
        'it','its','who','whom','which','what','when','where','why','how',

        -- auxiliaries / modality
        'will','would','should','could','can','may','might','must',
        'have','has','had','having','do','does','did','doing',

        -- very common verbs with low topical signal
        'use','using','used','make','makes','making','get','gets','getting',
        'help','helps','helping','provide','provides','providing',
        'support','supports','supporting','need','needs','needing',
        'allow','allows','allowing','enable','enables','enabling',

        -- DonorsChoose / proposal boilerplate
        'project','students','student','classroom','learning','school',
        'teacher','teachers','education','educational','grade','grades',
        'materials','supplies','tools','resources',
        'funded','funding','donors','donor','helpful','important',
        'essential','basic','provide','providing','supportive',

        -- vague evaluative language
        'better','best','more','most','many','much','very','really',
        'great','huge','positive','effective','meaningful',

        -- time / frequency
        'day','days','week','weeks','year','years','today','currently',
        'often','always','sometimes','usually',

        -- misc essay fillers
        'also','just','even','still','already','rather','quite',
        'each','every','both','either','neither')
GROUP BY project_id;
