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
        AND project_workflow_facts.last_content_or_resource_approved_at >= CONVERT_TIMEZONE('America/New_York', 'UTC', TIMESTAMP '2014-07-01')
        AND project.project_id > 0
        AND project.project_id % 50 = 1
),
numbers AS (
    SELECT ROW_NUMBER() OVER ()::INT AS n
    FROM dbt_target.fct_projects
    LIMIT 500
),
token_positions AS (
    SELECT
        bp.project_id,
        n.n AS position,
        LOWER(
            SPLIT_PART(
                REGEXP_REPLACE(bp.essay, '[^a-zA-Z]+', ' '),
                ' '::VARCHAR,
                n.n
            )
        ) AS token
    FROM base_projects bp
    JOIN numbers n
        ON n.n <= REGEXP_COUNT(
            REGEXP_REPLACE(bp.essay, '[^a-zA-Z]+', ' '),
            ' '
        ) + 1
    WHERE bp.essay IS NOT NULL
)
SELECT
    tp.project_id,
    LISTAGG(tp.token, ',') WITHIN GROUP (ORDER BY tp.position ASC) AS tokens
FROM (
    SELECT *,
        SUM(CASE WHEN rn_per_project_token = 1 THEN 1 ELSE 0 END)
            OVER (PARTITION BY token) AS doc_count
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY project_id, token ORDER BY position) AS rn_per_project_token
        FROM token_positions
    )
) tp
WHERE
    -- term is removed if it appears in <25 projects
    tp.doc_count >= 25
    -- only terms 3+ characters or 'ai'
    AND (LENGTH(token) >= 3 OR token = 'ai')

    -- static stopword removal
    AND token NOT IN 
    -- articles / conjunctions / prepositions
        ('the','and','for','with','are','was','were','been','being',
        'that','this','these','those','from','into','about','over',
        'under','after','before','between','while','than','then',
        'because','through','during','without','within',
        'but','not','nor','yet','all','any','per','via','out','off','due','own','such',

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
        'enable','enables','enabling',
        'give','gives','giving','given','want','wants','know','knows',
        'take','takes','taking','come','comes','look','looks','looking',
        'try','tries','include','includes','including',

        -- DonorsChoose / proposal boilerplate
        'teaching','lesson','lessons','impact',
        'grant','grants',

        -- vague evaluative language
        'better','best','more','most','many','much','very','really',
        'great','huge','positive','effective','meaningful',
        'good','well','new','real','different','little','few','long',
        'high','large','small','young','old','possible',
        'truly','simply','especially','particularly','actually','clearly','obviously',

        -- time / frequency
        'day','days','week','weeks','year','years','today','currently',
        'often','always','sometimes','usually',
        'time','times','month','months','annual','weekly','daily','monthly',

        -- misc essay fillers
        'also','just','even','still','already','rather','quite',
        'each','every','both','either','neither',
        'like','too','though','however','therefore','thus','indeed',
        'specifically','generally','typically','basically','literally')
    GROUP BY tp.project_id;