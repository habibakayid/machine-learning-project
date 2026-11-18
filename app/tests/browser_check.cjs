const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");
const base = path.resolve(__dirname,"..");
const fixtures = JSON.parse(fs.readFileSync(path.join(base,"tests/fixtures.json")));
(async () => {
  const browser = await chromium.launch({headless:true,executablePath:process.env.FLOWERS_CHROMIUM || undefined});
  const errors = [];
  const results = [];
  try {
    const page = await browser.newPage({viewport:{width:1280,height:940}});
    page.on("pageerror",e=>errors.push(String(e)));
    page.on("console",e=>{if(e.type()==="error") errors.push(e.text());});
    await page.goto("http://127.0.0.1:8510/");
    await page.getByRole("heading",{name:"Flowers Classification"}).waitFor({timeout:120000});
    assert.equal(await page.title(),"Flowers Classification");
    assert((await page.getByRole("combobox",{name:"Training data",exact:true}).inputValue()).includes("Kaggle only"));
    const scopeSelect=page.getByRole("combobox",{name:"Training data",exact:true});
    const modelSelect=page.getByRole("combobox",{name:"Model",exact:true});
    const upload=page.locator('input[type=file]');
    for (const scope of ["primary"]) {
      await scopeSelect.click();
      await page.getByRole("option",{name:scope==="primary"?"Kaggle only (primary)":"Kaggle + field (supplemental)",exact:true}).click();
      for (const family of ["Logistic Regression","KNN Classifier","K-Means"]) {
        await modelSelect.click();
        await page.getByRole("option",{name:family,exact:true}).click();
        for (const f of fixtures.filter(v=>v.scope===scope&&v.family===family)) {
          await upload.setInputFiles(path.join(base,f.path));
          await page.locator('[data-testid="stImage"]').getByText(path.basename(f.path),{exact:true}).waitFor({timeout:45000});
          await page.getByText("Prediction · "+family,{exact:true}).waitFor();
          await page.getByRole("heading",{name:f.predicted[0].toUpperCase()+f.predicted.slice(1),exact:true}).waitFor();
          if (f.probabilities) {
            await page.locator('[data-testid="stProgress"]').nth(4).waitFor();
            await page.waitForFunction(expected=>{
              const values=Array.from(document.querySelectorAll("[data-testid=stProgress]"),el=>el.textContent);
              return values.length===5&&Object.entries(expected).every(([label,p])=>values.some(text=>{
                const number=text.match(/([0-9.]+)%/);
                return text.toLowerCase().includes(label)&&number&&Math.abs(Number(number[1])-100*p)<=.051;
              }));
            },f.probabilities,{timeout:20000});
          } else {
            await page.getByText("Cluster "+f.cluster_display+" was assigned this label",{exact:false}).waitFor();
            assert.equal(await page.locator('[data-testid="stProgress"]').count(),0);
          }
          assert.equal(await page.locator('[data-testid="stException"]').count(),0);
          results.push({scope,family,actual:f.actual,predicted:f.predicted,result:"passed"});
          console.log(scope,family,f.actual,"->",f.predicted);
        }
        await page.screenshot({path:path.join(process.env.FLOWERS_QA_FOLDER,"optimization-"+scope+"-"+family.replaceAll(" ","_")+".png")});
      }
    }
    await upload.setInputFiles({name:"corrupt.jpg",mimeType:"image/jpeg",buffer:Buffer.from("invalid photo")});
    await page.getByText("This image could not be opened.",{exact:false}).waitFor();
    await upload.setInputFiles({name:"unsupported.jpg",mimeType:"image/jpeg",
      buffer:Buffer.from("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7","base64")});
    await page.getByText("This image format is unsupported.",{exact:false}).waitFor();
    await upload.setInputFiles(path.join(base,fixtures.at(-1).path));
    await page.getByText("Prediction · K-Means",{exact:true}).waitFor();
    const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    await mobile.goto("http://127.0.0.1:8510/");
    await mobile.getByRole("button",{name:"Try a dataset example"}).click();
    await mobile.getByText("Estimated class probabilities").waitFor({timeout:45000});
    assert(await mobile.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    await mobile.screenshot({path:path.join(process.env.FLOWERS_QA_FOLDER,"optimization-mobile.png"),fullPage:true});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(base,"evidence/BROWSER_VERIFICATION.json"),JSON.stringify({
      url:"http://127.0.0.1:8510/",browser:"Chromium / Playwright",desktop:[1280,940],mobile:[390,844],
      browser_plugin:"not available",cases:results,corrupt_upload:"passed",unsupported_format:"passed",
      recovery:"passed",console_errors:errors,mobile_layout:"passed"
    },null,2));
    console.log("PASS",results.length,"model/class uploads, predictions, probabilities/cluster, errors and mobile.");
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});



